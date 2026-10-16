import sys
import os; sys.path.insert(0, os.environ.get("AMPHION_PATH", "./Amphion"))

import glob, json, os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio, librosa, soundfile as sf
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, WavLMForXVector
from models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from fsq_factors import FSQFactor

N, MAX_SEC = 500, 12
dev = "cuda"
random.seed(0)

# --- Same 500 sentences as eval_curve.py / test_wer.py
root = "./data/LibriSpeech/test-clean"
items = []
for trans in glob.glob(f"{root}/*/*/*.trans.txt"):
    d = os.path.dirname(trans)
    for line in open(trans):
        uid, text = line.strip().split(" ", 1)
        items.append((f"{d}/{uid}.flac", text, uid.split("-")[0]))
random.shuffle(items)
sel = [it for it in items if sf.info(it[0]).duration <= MAX_SEC][:N]

# --- Frozen FACodec
enc = FACodecEncoder(ngf=32, up_ratios=[2, 4, 5, 5], out_channels=256)
dec = FACodecDecoder(
    in_channels=256, upsample_initial_channel=1024, ngf=32,
    up_ratios=[5, 5, 4, 2], vq_num_q_c=2, vq_num_q_p=1, vq_num_q_r=3,
    vq_dim=256, codebook_dim=8, codebook_size_prosody=10,
    codebook_size_content=10, codebook_size_residual=10,
    use_gr_x_timbre=True, use_gr_residual_f0=True, use_gr_residual_phone=True,
)
repo = "amphion/naturalspeech3_facodec"
enc.load_state_dict(torch.load(hf_hub_download(repo, "ns3_facodec_encoder.bin")))
dec.load_state_dict(torch.load(hf_hub_download(repo, "ns3_facodec_decoder.bin")))
enc.eval().to(dev); dec.eval().to(dev)

# --- All trained FSQ models found on disk (fsq_<name>.json / .pt)
def levels_for_bits(bits):
    return [4] * (bits // 2) + ([2] if bits % 2 else [])

fsq = {}
for jf in sorted(glob.glob("fsq_*.json")):
    name = jf[len("fsq_"):-len(".json")]
    if not os.path.exists(f"fsq_{name}.pt"):
        continue
    cfg = json.load(open(jf))
    bits = [cfg["p"], cfg["c"]] + ([cfg["r"]] if cfg["r"] > 0 else [])
    m = nn.ModuleList([FSQFactor(levels_for_bits(b)) for b in bits]).to(dev)
    m.load_state_dict(torch.load(f"fsq_{name}.pt")); m.eval()
    fsq[name] = (m, bits)
print("FSQ models found:", {n: sum(b) * 80 / 1000 for n, (_, b) in fsq.items()}, "kbps")

# --- Truncated RVQ: sum of the first k levels of each factor
def layers_of(q):
    return q.layers if hasattr(q, "layers") else q.quantizers

def rvq_emb(ids, kc, kr):                         # ids: [6, B, T] = 1 prosody, 2 content, 3 residual
    qp, qc, qr = dec.quantizer[0], dec.quantizer[1], dec.quantizer[2]
    emb = layers_of(qp)[0].vq2emb(ids[0])
    for i in range(kc):
        emb = emb + layers_of(qc)[i].vq2emb(ids[1 + i])
    for i in range(kr):
        emb = emb + layers_of(qr)[i].vq2emb(ids[3 + i])
    return emb

# condition name -> (family, kbps)
RATE = {"rvq_full": ("RVQ", 4.8), "rvq_nores": ("RVQ", 2.4), "rvq_p1c1": ("RVQ", 1.6)}
for n, (_, bits) in fsq.items():
    if len(bits) == 3:
        RATE[f"fsq_{n}"] = ("FSQ", 80 * sum(bits) / 1000)
        RATE[f"fsq_{n}_nores"] = ("FSQ", 80 * sum(bits[:2]) / 1000)
    else:
        RATE[f"fsq_{n}"] = ("FSQ", 80 * sum(bits) / 1000)

# --- Metrics: log-mel L1 and speaker similarity (WavLM x-vector, cosine)
mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=1024, hop_length=256, n_mels=80).to(dev)
def mel_l1(x, y):
    n = min(x.shape[-1], y.shape[-1])
    lm = lambda t: torch.log(mel(t[..., :n].squeeze(1)) + 1e-5)
    return F.l1_loss(lm(x), lm(y)).item()

fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
@torch.no_grad()
def spk_emb(wav):                                  # wav: tensor [1,1,T] at 16 kHz
    inp = fe(wav[0, 0].cpu().numpy(), sampling_rate=16000, return_tensors="pt")
    return F.normalize(sv(**{k: v.to(dev) for k, v in inp.items()}).embeddings, dim=-1)[0]

# --- Run
res = {k: {"mel": [], "sim": []} for k in RATE}
orig_emb, spk_ids = [], []
for i, (path, _, spk_id) in enumerate(sel):
    y, _ = librosa.load(path, sr=16000)
    wav = torch.from_numpy(y).float()[None, None, :].to(dev)
    e_orig = spk_emb(wav); orig_emb.append(e_orig); spk_ids.append(spk_id)
    with torch.no_grad():
        h = enc(wav)
        _, ids, _, _, spk = dec(h, vq=True)
        outs = {"rvq_full": dec.inference(rvq_emb(ids, 2, 3), spk),
                "rvq_nores": dec.inference(rvq_emb(ids, 2, 0), spk),
                "rvq_p1c1": dec.inference(rvq_emb(ids, 1, 0), spk)}
        for n, (m, bits) in fsq.items():
            p, _ = m[0](h); c, _ = m[1](h)
            if len(bits) == 3:
                r, _ = m[2](h - (p + c))
                outs[f"fsq_{n}"] = dec.inference(p + c + r, spk)
                outs[f"fsq_{n}_nores"] = dec.inference(p + c, spk)
            else:
                outs[f"fsq_{n}"] = dec.inference(p + c, spk)
    for k, out in outs.items():
        res[k]["mel"].append(mel_l1(out, wav))
        res[k]["sim"].append(float(e_orig @ spk_emb(out)))
    print(f"{i + 1}/{N}", end="\r")

json.dump(res, open("fsq_extra_metrics.json", "w"))

# --- Reference points for the similarity scale (from original audio only)
E_ = torch.stack(orig_emb).cpu(); S = (E_ @ E_.T).numpy()
same = np.array([[spk_ids[a] == spk_ids[b] and a != b for b in range(N)] for a in range(N)])
diff_spk = np.array([[spk_ids[a] != spk_ids[b] for b in range(N)] for a in range(N)])
print(f"\nreference similarity: same speaker, other utterance {S[same].mean():.3f} | different speakers {S[diff_spk].mean():.3f}")

# --- Statistics (bootstrap over utterances)
idx = np.random.default_rng(0).integers(0, N, size=(2000, N))
ci = lambda v: f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"
print(f"\n{'condition':18s} {'family':>6s} {'kbps':>5s}  {'log-mel L1 (lower=better)':>28s}  {'speaker sim (higher=better)':>30s}")
for k, (fam, kbps) in sorted(RATE.items(), key=lambda kv: (kv[1][1], kv[1][0])):
    a, b = np.array(res[k]["mel"]), np.array(res[k]["sim"])
    print(f"{k:18s} {fam:>6s} {kbps:5.1f}  {a.mean():7.3f} {ci(a[idx].mean(1)):>20s}  {b.mean():7.3f} {ci(b[idx].mean(1)):>20s}")

# --- Paired differences FSQ - RVQ at the same bitrate
print("\npaired differences at matched bitrate (FSQ - RVQ), 95% CI over utterances:")
for k, (fam, kbps) in sorted(RATE.items(), key=lambda kv: kv[1][1]):
    if fam != "FSQ":
        continue
    ref = [r for r, (f, kb) in RATE.items() if f == "RVQ" and abs(kb - kbps) < 1e-6]
    if not ref:
        continue
    r = ref[0]
    dm = (np.array(res[k]["mel"]) - np.array(res[r]["mel"]))[idx].mean(1)
    ds = (np.array(res[k]["sim"]) - np.array(res[r]["sim"]))[idx].mean(1)
    print(f"{k:18s} vs {r:10s} ({kbps:.1f} kbps): mel {dm.mean():+.3f} {ci(dm)} | speaker sim {ds.mean():+.3f} {ci(ds)}")