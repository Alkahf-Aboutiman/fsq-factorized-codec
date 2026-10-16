import sys
import os; sys.path.insert(0, os.environ.get("AMPHION_PATH", "./Amphion"))

import glob, pickle, random
import numpy as np, torch, torch.nn.functional as F
import librosa, torchaudio
from scipy.signal import fftconvolve
from huggingface_hub import hf_hub_download
from models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from env_models import EnvEncoder, Renderer

CKPT, N_UTT, SEC = "env_model.pt", 300, 4
dev = "cuda"
random.seed(0); rng = np.random.default_rng(0)

files = sorted(glob.glob("./data/LibriSpeech/test-clean/*/*/*.flac"))
random.shuffle(files)
def crop(path):
    y = librosa.load(path, sr=16000)[0].astype(np.float64)
    n = SEC * 16000
    y = np.pad(y, (0, max(0, n - len(y))))
    s = random.randint(0, len(y) - n)
    return y[s:s + n]
def spk_of(path):
    return path.split("/")[-3]

rooms = pickle.load(open("rir_bank.pkl", "rb"))["test"]          # unseen rooms
rts = np.array([r[1] for r in rooms])

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

w = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float()[None, None].to(dev)
mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=1024, hop_length=256, n_mels=80).to(dev)
def mel_l1(x, y):
    n = min(x.shape[-1], y.shape[-1])
    lm = lambda t: torch.log(mel(t[..., :n].squeeze(1)) + 1e-5)
    return F.l1_loss(lm(x), lm(y)).item()
def reverb_only(sig, rir):
    y = fftconvolve(sig, rir)[: len(sig)]
    return y * (np.sqrt(np.mean(sig ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-9))

# signed difference d_rt = RT60(room of e) - RT60(true room); same room is handled separately
BINS = [("e much shorter (< -0.3 s)", -9, -0.3), ("e shorter (-0.3..-0.1)", -0.3, -0.1),
        ("e slightly different (|d|<0.1)", -0.1, 0.1), ("e longer (+0.1..+0.3)", 0.1, 0.3),
        ("e much longer (> +0.3 s)", 0.3, 9)]

rows = []          # (utterance idx, bin name or "same", d_rt, mel distance)
for i in range(N_UTT):
    path = files[i]; Y = crop(path)
    k = rng.integers(len(rooms)); rir_k, rt_k = rooms[k]
    dY = reverb_only(Y, rir_k)                                   # ground truth: Y in the true room
    with torch.no_grad():
        h = enc(w(Y)); _, ids, _, _, spk = dec(h, eval_vq=False, vq=True)
        emb_c = dec.vq2emb(ids, use_residual_code=False)

    def dist(j):
        """Render Y with e extracted from ANOTHER speaker reverberated in room j."""
        other = random.choice([f for f in files[N_UTT:] if spk_of(f) != spk_of(path)])
        with torch.no_grad():
            e, _ = E(enc(w(reverb_only(crop(other), rooms[j][0]))))
            return mel_l1(dec.inference(R(emb_c, e), spk), w(dY))

    rows.append((i, "same", 0.0, dist(k)))
    for name, lo, hi in BINS:
        cand = [j for j in range(len(rooms)) if j != k and lo <= rts[j] - rt_k < hi]
        if cand:
            j = random.choice(cand)
            rows.append((i, name, rts[j] - rt_k, dist(j)))
    print(f"{i + 1}/{N_UTT}", end="\r")

# --- Statistics: paired against the same-room render of the same utterance
same = {r[0]: r[3] for r in rows if r[1] == "same"}
bs = np.random.default_rng(0)
print(f"\n{'e source':32s} {'n':>4s} {'mel dist':>9s} {'vs same room':>13s}   95% CI           e better than... (share d<same)")
print(f"{'same room (other speaker)':32s} {len(same):4d} {np.mean(list(same.values())):9.3f}")
for name, _, _ in BINS:
    sub = [(r[0], r[3]) for r in rows if r[1] == name]
    if not sub:
        continue
    dd = np.array([d - same[i] for i, d in sub])
    boot = [bs.choice(dd, len(dd)).mean() for _ in range(2000)]
    print(f"{name:32s} {len(sub):4d} {np.mean([d for _, d in sub]):9.3f} {dd.mean():+13.3f}   "
          f"[{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]   {100 * (dd < 0).mean():.0f}%")

# --- Trend: does the distance grow with |d_rt|?
nz = [(abs(r[2]), r[3] - same[r[0]]) for r in rows if r[1] != "same"]
a, b = np.array(nz).T
print(f"\ncorrelation between |RT60 difference| and extra log-mel distance: {np.corrcoef(a, b)[0, 1]:+.2f}  "
      f"(slope {np.polyfit(a, b, 1)[0]:+.3f} per second)")
