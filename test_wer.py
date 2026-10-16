import sys
import os; sys.path.insert(0, os.environ.get("AMPHION_PATH", "./Amphion"))

import glob, os, random, re
import torch, torch.nn.functional as F
import librosa, soundfile as sf, torchaudio
from huggingface_hub import hf_hub_download
from transformers import pipeline
from models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
import numpy as np, json

N, MAX_SEC = 500, 12
dev = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0)

# --- Data: LibriSpeech test-clean (~350 MB on first run)
torchaudio.datasets.LIBRISPEECH("./data", url="test-clean", download=True)
root = "./data/LibriSpeech/test-clean"
items = []
for trans in glob.glob(f"{root}/*/*/*.trans.txt"):
    d = os.path.dirname(trans)
    for line in open(trans):
        uid, text = line.strip().split(" ", 1)
        items.append((f"{d}/{uid}.flac", text, uid.split("-")[0]))
random.shuffle(items)
sel = [it for it in items if sf.info(it[0]).duration <= MAX_SEC][:N]

# --- FACodec (frozen)
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

# --- FSQ factors, config A: 10 + 20 + 30 bits per frame = 4.8 kbps (2.4 kbps without residual)
from fsq_factors import FSQFactor
facs = torch.nn.ModuleList([FSQFactor([4] * 5), FSQFactor([4] * 10), FSQFactor([4] * 15)]).to(dev)
facs.load_state_dict(torch.load("fsq_A.pt")); facs.eval()

@torch.no_grad()
def fsq_decode(wav, use_res=True):
    h = enc(wav)
    spk = dec(h, vq=True)[-1]
    p, _ = facs[0](h); c, _ = facs[1](h)
    emb = p + c
    if use_res:
        r, _ = facs[2](h - (p + c))
        emb = emb + r
    return dec.inference(emb, spk)[0, 0].cpu().numpy()

def to_wav(path):
    y, _ = librosa.load(path, sr=16000)
    return torch.from_numpy(y).float()[None, None, :].to(dev)

@torch.no_grad()
def encode(wav):
    _, ids, _, _, spk = dec(enc(wav), eval_vq=False, vq=True)
    return ids, spk          # ids: [prosody 1, content 2, residual 3, B, T]

@torch.no_grad()
def decode(ids, spk, use_residual=True):
    emb = dec.vq2emb(ids, use_residual_code=use_residual)
    return dec.inference(emb, spk)[0, 0].cpu().numpy()

# --- ASR
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small",
               device=0 if dev == "cuda" else -1)

def norm(t):
    return " ".join(re.sub(r"[^a-z' ]", "", t.lower()).split())

def transcribe(y):
    out = asr({"raw": y, "sampling_rate": 16000}, generate_kwargs={"language": "en"})
    return norm(out["text"])

# --- Experiment
conds = {k: [] for k in ["orig", "recon", "recon_nores", "timbre_swap", "prosody_swap", "fsq", "fsq_nores"]}
refs = []
for i, (path, text, spk) in enumerate(sel):
    donor = random.choice([s for s in sel if s[2] != spk])
    wa, wb = to_wav(path), to_wav(donor[0])
    ids_a, spk_a = encode(wa)
    ids_b, spk_b = encode(wb)

    # prosody of B stretched to the length of A (coarse alignment, nearest neighbour)
    pros_b = F.interpolate(ids_b[:1].float(), size=ids_a.shape[-1], mode="nearest").long()
    mix = torch.cat([pros_b, ids_a[1:]], dim=0)

    outs = {
        "orig": wa[0, 0].cpu().numpy(),
        "recon": decode(ids_a, spk_a),
        "recon_nores": decode(ids_a, spk_a, False),
        "timbre_swap": decode(ids_a, spk_b, False),
        "prosody_swap": decode(mix, spk_a, False),
        "fsq": fsq_decode(wa),
        "fsq_nores": fsq_decode(wa, False),
    }
    refs.append(norm(text))
    for k, y in outs.items():
        conds[k].append(transcribe(y))
    print(f"{i+1}/{len(sel)}", end="\r")

json.dump({"refs": refs, **conds}, open("wer_hyps.json", "w"))   # save the transcriptions

def word_errors(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[-1]

nref = np.array([len(r.split()) for r in refs])
errs = {k: np.array([word_errors(r, h) for r, h in zip(refs, hyps)]) for k, hyps in conds.items()}

rng = np.random.default_rng(0)
idx = rng.integers(0, len(refs), size=(2000, len(refs)))      # bootstrap over sentences

print(f"\n{'condition':15s} {'WER':>6s}   95% CI")
for k, e in errs.items():
    w = 100 * e[idx].sum(1) / nref[idx].sum(1)
    lo, hi = np.percentile(w, [2.5, 97.5])
    print(f"{k:15s} {100 * e.sum() / nref.sum():5.1f}%   [{lo:.1f}, {hi:.1f}]")

def diff(a, b):
    d = 100 * (errs[a][idx].sum(1) - errs[b][idx].sum(1)) / nref[idx].sum(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"{a} - {b}: {d.mean():+.2f} pt   95% CI [{lo:+.2f}, {hi:+.2f}]")

print()
diff("fsq", "recon")
diff("fsq_nores", "recon_nores")
diff("timbre_swap", "recon_nores")
diff("prosody_swap", "recon_nores")