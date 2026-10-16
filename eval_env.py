import sys
import os; sys.path.insert(0, os.environ.get("AMPHION_PATH", "./Amphion"))

import glob, os, pickle, random, re
import numpy as np, torch, torch.nn.functional as F
import librosa, soundfile as sf, torchaudio
from scipy.signal import fftconvolve
from huggingface_hub import hf_hub_download
from transformers import pipeline
from models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from env_models import EnvEncoder, Renderer, NOISES
from degrade import degrade

N, MAX_SEC, CKPT = 100, 12, "env_model.pt"
SNR_RANGE = (10, 30)                       # milder than training (0-20 dB)
SCENARIOS = ["reverb_only", "noise_only", "both"]
dev = "cuda"
random.seed(0); rng = np.random.default_rng(0)

# --- test-clean utterances
root = "./data/LibriSpeech/test-clean"
items = []
for trans in glob.glob(f"{root}/*/*/*.trans.txt"):
    d = os.path.dirname(trans)
    for line in open(trans):
        uid, text = line.strip().split(" ", 1)
        items.append((f"{d}/{uid}.flac", text, uid.split("-")[0]))
random.shuffle(items)
items = [it for it in items if sf.info(it[0]).duration <= MAX_SEC]
sel = items[:N]
pool = [sf.read(it[0], dtype="float32")[0] for it in items[N:N + 40]]

# --- Models
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
ck = torch.load(CKPT)
E, R = EnvEncoder().to(dev), Renderer().to(dev)
E.load_state_dict(ck["E"]); R.load_state_dict(ck["R"]); E.eval(); R.eval()
rooms = pickle.load(open("rir_bank.pkl", "rb"))["test"]          # unseen rooms

mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=1024, hop_length=256, n_mels=80).to(dev)
def mel_l1(x, y):
    n = min(x.shape[-1], y.shape[-1])
    lm = lambda t: torch.log(mel(t[..., :n].squeeze(1)) + 1e-5)
    return F.l1_loss(lm(x), lm(y)).item()

w = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float()[None, None].to(dev)
to_np = lambda t: t[0, 0].cpu().numpy()

@torch.no_grad()
def lat(wav, full):
    h = enc(wav)
    _, ids, _, _, spk = dec(h, eval_vq=False, vq=True)
    return h, dec.vq2emb(ids, use_residual_code=full), spk

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0)
norm = lambda t: " ".join(re.sub(r"[^a-z' ]", "", t.lower()).split())
def transcribe(y):
    out = asr({"raw": np.asarray(y, dtype=np.float32), "sampling_rate": 16000}, generate_kwargs={"language": "en"})
    return norm(out["text"])

def make_deg(sig, scenario, rir, rt60, snr, noise):
    """Apply only the degradations of the chosen scenario."""
    if scenario == "reverb_only":
        y = fftconvolve(sig, rir)[: len(sig)]
        return y * (np.sqrt(np.mean(sig ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-9))
    if scenario == "noise_only":
        return degrade(sig, rng, pool, rt60=rt60, snr=snr, noise=noise, rir=np.array([1.0]))[0]
    return degrade(sig, rng, pool, rt60=rt60, snr=snr, noise=noise, rir=rir)[0]

def word_errors(ref, hyp):
    r, h = ref.split(), hyp.split()
    dd = list(range(len(h) + 1))
    for a, rw in enumerate(r, 1):
        prev, dd[0] = dd[0], a
        for b, hw in enumerate(h, 1):
            prev, dd[b] = dd[b], min(dd[b] + 1, dd[b - 1] + 1, prev + (rw != hw))
    return dd[-1]

# --- Experiment: same sentences for every scenario
refs = [norm(t) for _, t, _ in sel]
nref = np.array([len(r.split()) for r in refs])
hyps = {"orig": [], "clean_codec": []}
res = {s: {"true_degraded": [], "render": [], "render_wrong_room": [], "d_ok": [], "d_bad": [],
           "noise": [], "snr": [], "rt60": []} for s in SCENARIOS}

for i, (path, text, spk) in enumerate(sel):
    Y = librosa.load(path, sr=16000)[0].astype(np.float64)
    xp = random.choice([s for s in sel if s[2] != spk])
    X = librosa.load(xp[0], sr=16000)[0].astype(np.float64)
    (rir, rt60), (rir_w, rt60_w) = random.sample(rooms, 2)
    snr, noise = rng.uniform(*SNR_RANGE), NOISES[rng.integers(len(NOISES))]
    with torch.no_grad():
        _, emb_c, s = lat(w(Y), False)
        _, emb_cf, _ = lat(w(Y), True)
        clean = to_np(dec.inference(emb_cf, s))
    hyps["orig"].append(transcribe(Y)); hyps["clean_codec"].append(transcribe(clean))
    for sc in SCENARIOS:
        dY = make_deg(Y, sc, rir, rt60, snr, noise)        # ground truth
        dX = make_deg(X, sc, rir, rt60, snr, noise)        # e source, same room
        dXw = make_deg(X, sc, rir_w, rt60_w, snr, noise)   # e source, WRONG room
        with torch.no_grad():
            e, _ = E(enc(w(dX))); e_w, _ = E(enc(w(dXw)))
            r, rw = dec.inference(R(emb_c, e), s), dec.inference(R(emb_c, e_w), s)
        wd = w(dY)
        o = res[sc]
        o["d_ok"].append(mel_l1(r, wd)); o["d_bad"].append(mel_l1(rw, wd))
        o["true_degraded"].append(transcribe(dY))
        o["render"].append(transcribe(to_np(r))); o["render_wrong_room"].append(transcribe(to_np(rw)))
        o["noise"].append(noise); o["snr"].append(snr); o["rt60"].append(rt60)
    print(f"{i + 1}/{N}", end="\r")

# --- Statistics (bootstrap over utterances)
idx = np.random.default_rng(0).integers(0, N, size=(2000, N))
errs = lambda hy: np.array([word_errors(r, h) for r, h in zip(refs, hy)])
ci = lambda v: f"[{np.percentile(v, 2.5):+.1f}, {np.percentile(v, 97.5):+.1f}]"
wer = lambda er: 100 * er.sum() / nref.sum()
werb = lambda er: 100 * er[idx].sum(1) / nref[idx].sum(1)
diffb = lambda a, b: 100 * (a[idx].sum(1) - b[idx].sum(1)) / nref[idx].sum(1)

e_orig, e_clean = errs(hyps["orig"]), errs(hyps["clean_codec"])
print(f"\norig {wer(e_orig):.1f}% | clean_codec {wer(e_clean):.1f}%  (Whisper-small, {N} sentences)")
print(f"\n{'scenario':12s} {'true_deg':>9s} {'render':>8s} {'wrong':>8s} | render-codec (pt)   wrong-render (pt)")
for sc in SCENARIOS:
    o = res[sc]
    er = {k: errs(o[k]) for k in ["true_degraded", "render", "render_wrong_room"]}
    d1 = diffb(er["render"], e_clean); d2 = diffb(er["render_wrong_room"], er["render"])
    print(f"{sc:12s} {wer(er['true_degraded']):8.1f}% {wer(er['render']):7.1f}% {wer(er['render_wrong_room']):7.1f}% | "
          f"{d1.mean():+6.1f} {ci(d1)}   {d2.mean():+6.1f} {ci(d2)}")

print("\nlog-mel distance to the TRUE degraded Y (right e vs wrong room):")
for sc in SCENARIOS:
    o = res[sc]; d_ok, d_bad = np.array(o["d_ok"]), np.array(o["d_bad"])
    gap = (d_bad - d_ok)[idx].mean(1)
    print(f"{sc:12s} right {d_ok.mean():.3f} | wrong {d_bad.mean():.3f} | right better in {100 * (d_ok < d_bad).mean():.0f}% | "
          f"gap {gap.mean():+.3f} [{np.percentile(gap, 2.5):+.3f}, {np.percentile(gap, 97.5):+.3f}]")

print("\nRender WER by RT60 tercile (reverb_only):")
rt = np.array(res["reverb_only"]["rt60"]); er_r = errs(res["reverb_only"]["render"])
edges = np.percentile(rt, [0, 33.3, 66.6, 100])
edges[-1] += 1e-6                                   # make the last bin inclusive
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (rt >= lo) & (rt < hi)
    print(f"RT60 {lo:.2f}-{hi:.2f} s (n={m.sum()}): render {100 * er_r[m].sum() / nref[m].sum():.1f}% | "
          f"codec {100 * e_clean[m].sum() / nref[m].sum():.1f}%")

print("\nRender WER by noise type (both):")
nz = np.array(res["both"]["noise"]); er_b = errs(res["both"]["render"])
for n in sorted(set(nz)):
    m = nz == n
    print(f"{str(n):8s}: render {100 * er_b[m].sum() / nref[m].sum():.1f}% | "
          f"codec {100 * e_clean[m].sum() / nref[m].sum():.1f}%  (n={m.sum()})")

# per-room log-mel gap, to see whether a few rooms drive the average
rt_all = np.array(res["reverb_only"]["rt60"])
gap_i = np.array(res["reverb_only"]["d_bad"]) - np.array(res["reverb_only"]["d_ok"])
print("\nlog-mel gap (wrong - right) by RT60 tercile (reverb_only):")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (rt_all >= lo) & (rt_all < hi)
    print(f"RT60 {lo:.2f}-{hi:.2f} s: mean gap {gap_i[m].mean():+.3f} | right better in {100 * (gap_i[m] > 0).mean():.0f}%")