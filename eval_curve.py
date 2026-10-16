import sys
import os; sys.path.insert(0, os.environ.get("AMPHION_PATH", "./Amphion"))

import glob, json, os, random, re
import numpy as np
import torch, torch.nn as nn
import librosa, soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from transformers import pipeline
from models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from fsq_factors import FSQFactor

N, MAX_SEC = 500, 12
NEW_FSQ = ["B", "C", "D"]                     # FSQ configs to evaluate (fsq_<name>.pt / .json)
dev = "cuda"
random.seed(0)

def norm(t):
    return " ".join(re.sub(r"[^a-z' ]", "", t.lower()).split())

# --- Same 500 sentences as test_wer.py
root = "./data/LibriSpeech/test-clean"
items = []
for trans in glob.glob(f"{root}/*/*/*.trans.txt"):
    d = os.path.dirname(trans)
    for line in open(trans):
        uid, text = line.strip().split(" ", 1)
        items.append((f"{d}/{uid}.flac", text, uid.split("-")[0]))
random.shuffle(items)
sel = [it for it in items if sf.info(it[0]).duration <= MAX_SEC][:N]

old = json.load(open("wer_hyps.json"))
refs = [norm(t) for _, t, _ in sel]
assert refs == old["refs"], "Sentences do not match wer_hyps.json"
conds = {k: v for k, v in old.items() if k != "refs"}

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

# --- Trained FSQ models
def levels_for_bits(bits):
    return [4] * (bits // 2) + ([2] if bits % 2 else [])

fsq = {}
for name in NEW_FSQ:
    cfg = json.load(open(f"fsq_{name}.json"))
    bits = [cfg["p"], cfg["c"]] + ([cfg["r"]] if cfg["r"] > 0 else [])
    m = nn.ModuleList([FSQFactor(levels_for_bits(b)) for b in bits]).to(dev)
    m.load_state_dict(torch.load(f"fsq_{name}.pt")); m.eval()
    fsq[name] = (m, 80 * sum(bits) / 1000)            # (model, kbps)

# --- Truncated RVQ: sum of the first k levels of each factor
def layers_of(q):
    return q.layers if hasattr(q, "layers") else q.quantizers

def rvq_emb(ids, kc, kr):                             # ids: [6, B, T] = 1 prosody, 2 content, 3 residual
    qp, qc, qr = dec.quantizer[0], dec.quantizer[1], dec.quantizer[2]
    emb = layers_of(qp)[0].vq2emb(ids[0])
    for i in range(kc):
        emb = emb + layers_of(qc)[i].vq2emb(ids[1 + i])
    for i in range(kr):
        emb = emb + layers_of(qr)[i].vq2emb(ids[3 + i])
    return emb

# --- ASR
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0)

def transcribe(y):
    out = asr({"raw": y, "sampling_rate": 16000}, generate_kwargs={"language": "en"})
    return norm(out["text"])

def to_wav(path):
    y, _ = librosa.load(path, sr=16000)
    return torch.from_numpy(y).float()[None, None, :].to(dev)

# --- New conditions
new = {"rvq_p1c1": [], **{f"fsq_{n}": [] for n in NEW_FSQ}}
for i, (path, text, _) in enumerate(sel):
    wav = to_wav(path)
    with torch.no_grad():
        h = enc(wav)
        _, ids, _, _, spk = dec(h, vq=True)
        if i == 0:                                    # must reproduce the official vq2emb
            assert torch.allclose(rvq_emb(ids, 2, 3), dec.vq2emb(ids, use_residual_code=True),
                                  rtol=1e-3, atol=1e-3), "rvq_emb != vq2emb"
        outs = {"rvq_p1c1": dec.inference(rvq_emb(ids, 1, 0), spk)[0, 0].cpu().numpy()}
        for n, (m, _) in fsq.items():
            p, _ = m[0](h); c, _ = m[1](h)
            emb = p + c
            if len(m) == 3:
                r, _ = m[2](h - (p + c)); emb = emb + r
            outs[f"fsq_{n}"] = dec.inference(emb, spk)[0, 0].cpu().numpy()
    for k, y in outs.items():
        new[k].append(transcribe(y))
    print(f"{i + 1}/{len(sel)}", end="\r")

conds.update(new)
json.dump({"refs": refs, **conds}, open("wer_hyps_curve.json", "w"))

# --- Statistics (bootstrap over sentences)
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
idx = rng.integers(0, len(refs), size=(2000, len(refs)))

def wer_ci(k):
    e = errs[k]
    w = 100 * e[idx].sum(1) / nref[idx].sum(1)
    lo, hi = np.percentile(w, [2.5, 97.5])
    return 100 * e.sum() / nref.sum(), lo, hi

def diff(a, b):
    d = 100 * (errs[a][idx].sum(1) - errs[b][idx].sum(1)) / nref[idx].sum(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"{a} - {b}: {d.mean():+.2f} pt   95% CI [{lo:+.2f}, {hi:+.2f}]")

RATE = {                                              # condition -> (family, kbps)
    "recon": ("RVQ", 4.8), "recon_nores": ("RVQ", 2.4), "rvq_p1c1": ("RVQ", 1.6),
    "fsq": ("FSQ", 4.8), "fsq_nores": ("FSQ", 2.4),
}
for n, (_, kbps) in fsq.items():
    RATE[f"fsq_{n}"] = ("FSQ", kbps)

print(f"\n{'condition':12s} {'family':>7s} {'kbps':>5s} {'WER':>6s}   95% CI")
for k, (fam, kbps) in sorted(RATE.items(), key=lambda kv: (kv[1][1], kv[1][0])):
    w, lo, hi = wer_ci(k)
    print(f"{k:12s} {fam:>7s} {kbps:5.1f} {w:5.1f}%   [{lo:.1f}, {hi:.1f}]")
w, lo, hi = wer_ci("orig")
print(f"{'orig':12s} {'-':>7s} {'-':>5s} {w:5.1f}%   [{lo:.1f}, {hi:.1f}]\n")

diff("fsq_B", "rvq_p1c1")                             # 1.6 kbps
diff("fsq_nores", "recon_nores")                      # 2.4 kbps

# --- Rate vs WER curve
INK, GRID = "#2b2b2b", "#e3e3e0"
COL = {"RVQ": "#d9731a", "FSQ": "#2a6fdb"}
fig, ax = plt.subplots(figsize=(6.4, 4))
w0, lo0, hi0 = wer_ci("orig")
ax.axhspan(lo0, hi0, color=GRID, alpha=0.7, lw=0)
ax.axhline(w0, color="#8a8a85", ls="--", lw=1)
ax.text(0.65, w0 + 0.12, "original audio", color=INK, fontsize=8, va="bottom")
for fam in ("RVQ", "FSQ"):
    pts = sorted((kbps, k) for k, (f, kbps) in RATE.items() if f == fam)
    x = [p[0] for p in pts]
    st = [wer_ci(p[1]) for p in pts]
    y = [s[0] for s in st]
    yerr = [[s[0] - s[1] for s in st], [s[2] - s[0] for s in st]]
    ax.errorbar(x, y, yerr=yerr, color=COL[fam], lw=2, marker="o", ms=7, mec="white", mew=1.5,
                capsize=3, elinewidth=1.2, label=fam)
    ax.text(x[-1] + 0.08, y[-1], fam, color=INK, fontsize=9, va="center")
ax.set_xticks(sorted({v[1] for v in RATE.values()}))
ax.set_xlim(0.6, 5.3)
ax.set_xlabel("Bitrate (kbps)", color=INK)
ax.set_ylabel("Whisper-small WER (%)", color=INK)
ax.set_title("WER vs. bitrate (500 sentences, test-clean)", color=INK, fontsize=10, loc="left")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
os.makedirs("results", exist_ok=True)
fig.savefig("results/wer_vs_rate_unclipped.png", dpi=200)
print("figure: results/wer_vs_rate_unclipped.png")