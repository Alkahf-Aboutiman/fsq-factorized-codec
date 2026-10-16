import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("wer_hyps_curve.json"))
refs = d.pop("refs")
RATE = {                                              # condition -> (family, kbps)
    "rvq_p1c1": ("RVQ", 1.6), "recon_nores": ("RVQ", 2.4), "recon": ("RVQ", 4.8),
    "fsq_D": ("FSQ", 0.8), "fsq_C": ("FSQ", 1.2), "fsq_B": ("FSQ", 1.6),
    "fsq_nores": ("FSQ", 2.4), "fsq": ("FSQ", 4.8),
}

def word_errors(ref, hyp):
    r, h = ref.split(), hyp.split()
    dd = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, dd[0] = dd[0], i
        for j, hw in enumerate(h, 1):
            prev, dd[j] = dd[j], min(dd[j] + 1, dd[j - 1] + 1, prev + (rw != hw))
    return dd[-1]

nref = np.array([len(r.split()) for r in refs])
errs = {k: np.array([word_errors(r, h) for r, h in zip(refs, d[k])]) for k in list(RATE) + ["orig"]}
rng = np.random.default_rng(0)
idx = rng.integers(0, len(refs), size=(2000, len(refs)))      # bootstrap over sentences

def wer_ci(k):
    e = errs[k]
    w = 100 * e[idx].sum(1) / nref[idx].sum(1)
    lo, hi = np.percentile(w, [2.5, 97.5])
    return 100 * e.sum() / nref.sum(), lo, hi

# --- Diagnostic: do a few catastrophic sentences dominate the mean?
print(f"{'condition':12s} {'mean WER':>9s} {'median/sentence':>16s} {'sentences > 50 %':>17s}")
for k in ["orig"] + sorted(RATE, key=lambda k: RATE[k][1]):
    per = 100 * errs[k] / nref
    print(f"{k:12s} {100 * errs[k].sum() / nref.sum():8.1f}% {np.median(per):15.1f}% {100 * (per > 50).mean():16.1f}%")

# --- Rate vs WER curve
INK, GRID = "#2b2b2b", "#e3e3e0"
COL = {"RVQ": "#d9731a", "FSQ": "#2a6fdb"}
DX = {"RVQ": -0.07, "FSQ": 0.07}                      # small x offset so the markers do not hide each other
YMAX = 26
fig, ax = plt.subplots(figsize=(6.4, 4))
w0, lo0, hi0 = wer_ci("orig")
ax.axhspan(lo0, hi0, color=GRID, alpha=0.7, lw=0)
ax.axhline(w0, color="#8a8a85", ls="--", lw=1)
ax.text(0.65, hi0 + 0.2, "original audio", color=INK, fontsize=8, va="bottom")
for fam in ("RVQ", "FSQ"):
    pts = sorted((kbps, k) for k, (f, kbps) in RATE.items() if f == fam)
    x = [p[0] + DX[fam] for p in pts]
    st = [wer_ci(p[1]) for p in pts]
    y = [s[0] for s in st]
    yerr = [[s[0] - s[1] for s in st], [s[2] - s[0] for s in st]]
    ax.errorbar(x, y, yerr=yerr, color=COL[fam], lw=2, marker="o", ms=7, mec="white", mew=1.5,
                capsize=3, elinewidth=1.2, label=fam)
    ax.text(x[-1] + 0.25, y[-1], fam, color=INK, fontsize=9, va="center")
w_d = wer_ci("fsq_D")[0]
ax.text(1.5, YMAX - 0.4, f"FSQ 0.8 kbps: {w_d:.1f}% (off scale)", color=INK, fontsize=8, va="top")
ax.set_ylim(0, YMAX)
ax.set_xlim(0.6, 5.4)
ax.set_xticks(sorted({v[1] for v in RATE.values()}))
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
fig.savefig("results/wer_vs_rate.png", dpi=200)
print("figure: results/wer_vs_rate.png")