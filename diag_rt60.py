import sys
import os; sys.path.insert(0, os.environ.get("AMPHION_PATH", "./Amphion"))

import glob, os, pickle, random
import numpy as np, torch, torch.nn.functional as F
import librosa, soundfile as sf, torchaudio
from scipy.signal import fftconvolve
from huggingface_hub import hf_hub_download
from models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from env_models import EnvEncoder, Renderer, NOISES
from degrade import degrade

CKPT, N_UTT, SEC = "env_model.pt", 160, 4
dev = "cuda"
random.seed(0); rng = np.random.default_rng(0)

# --- Data: test-clean utterances, SEC-second crops
root = "./data/LibriSpeech/test-clean"
files = sorted(glob.glob(f"{root}/*/*/*.flac"))
random.shuffle(files)
def crop(path):
    y = librosa.load(path, sr=16000)[0].astype(np.float64)
    n = SEC * 16000
    y = np.pad(y, (0, max(0, n - len(y))))
    s = random.randint(0, len(y) - n)
    return y[s:s + n]
pool = [sf.read(f, dtype="float32")[0] for f in files[N_UTT * 2: N_UTT * 2 + 40]]   # babble source
rooms = pickle.load(open("rir_bank.pkl", "rb"))["test"]                              # unseen rooms

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

# scenario name -> function(sig, rir, rt60) -> degraded signal
SC = {
    "reverb_only (no noise)": lambda s, r, t: reverb_only(s, r),
    "noise 10-30 dB":         lambda s, r, t: degrade(s, rng, pool, rt60=t, snr=rng.uniform(10, 30),
                                                      noise=NOISES[rng.integers(len(NOISES))], rir=r)[0],
    "train-like 0-20 dB":     lambda s, r, t: degrade(s, rng, pool, rt60=t, snr=rng.uniform(0, 20),
                                                      noise=NOISES[rng.integers(len(NOISES))], rir=r)[0],
}

@torch.no_grad()
def embed(sig):
    e, aux = E(enc(w(sig)))
    return e[0].cpu(), float(aux[0, 0]) * 0.3 + 0.6          # e, predicted RT60 (s)

# --- 1) Does E recover RT60 on unseen rooms?
print("1) RT60 prediction by E on unseen rooms")
print(f"{'scenario':24s} {'RMSE (s)':>9s} {'corr':>6s} {'slope':>6s}   mean pred in RT60 terciles (short/mid/long)")
store = {}
for name, fn in SC.items():
    tr, pr, es, rid = [], [], [], []
    for i in range(N_UTT):
        k = rng.integers(len(rooms)); rir, rt60 = rooms[k]
        e, p = embed(fn(crop(files[i]), rir, rt60))
        tr.append(rt60); pr.append(p); es.append(e); rid.append(k)
    tr, pr = np.array(tr), np.array(pr)
    ed = np.quantile(tr, [0, 1 / 3, 2 / 3, 1]); ed[-1] += 1e-6
    ter = [pr[(tr >= lo) & (tr < hi)].mean() for lo, hi in zip(ed[:-1], ed[1:])]
    tt = [tr[(tr >= lo) & (tr < hi)].mean() for lo, hi in zip(ed[:-1], ed[1:])]
    print(f"{name:24s} {np.sqrt(np.mean((pr - tr) ** 2)):9.3f} {np.corrcoef(tr, pr)[0, 1]:6.2f} "
          f"{np.polyfit(tr, pr, 1)[0]:6.2f}   " + " / ".join(f"{a:.2f} (true {b:.2f})" for a, b in zip(ter, tt)))
print(f"   chance RMSE ~ {np.std(tr):.3f} s (always predicting the mean)")

# --- 2) Retrieval: do two different utterances in the SAME room give nearby e? (reverb only)
print("\n2) Room retrieval from e (reverb only, unseen rooms)")
n_r = len(rooms)
e1, e2 = [], []
for k, (rir, rt60) in enumerate(rooms):
    e1.append(embed(reverb_only(crop(random.choice(files)), rir))[0])
    e2.append(embed(reverb_only(crop(random.choice(files)), rir))[0])
e1, e2 = F.normalize(torch.stack(e1), dim=1), F.normalize(torch.stack(e2), dim=1)
rank = (e1 @ e2.T).argsort(1, descending=True)
top1 = np.mean([rank[k, 0].item() == k for k in range(n_r)])
top5 = np.mean([k in rank[k, :5].tolist() for k in range(n_r)])
rts = np.array([r[1] for r in rooms])
near = np.mean([abs(rts[rank[k, 0].item()] - rts[k]) < 0.1 for k in range(n_r)])
print(f"top-1 same room: {100 * top1:.0f}% (chance {100 / n_r:.1f}%) | top-5: {100 * top5:.0f}% | "
      f"retrieved room within 0.1 s of true RT60: {100 * near:.0f}%")

# --- 3) Does R actually use e? Render with e of a short room vs e of a long room
print("\n3) Sensitivity of the renderer to e")
order = np.argsort(rts); short, long_ = order[:10], order[-10:]
d_sl, d_clean_s, d_clean_l = [], [], []
for i in range(40):
    y = crop(files[i])
    with torch.no_grad():
        h = enc(w(y)); _, ids, _, _, spk = dec(h, eval_vq=False, vq=True)
        emb_c = dec.vq2emb(ids, use_residual_code=False)
        clean = dec.inference(dec.vq2emb(ids, use_residual_code=True), spk)
        es = E(enc(w(reverb_only(crop(random.choice(files)), rooms[random.choice(short)][0]))))[0]
        el = E(enc(w(reverb_only(crop(random.choice(files)), rooms[random.choice(long_)][0]))))[0]
        rs, rl = dec.inference(R(emb_c, es), spk), dec.inference(R(emb_c, el), spk)
    d_sl.append(mel_l1(rs, rl)); d_clean_s.append(mel_l1(rs, clean)); d_clean_l.append(mel_l1(rl, clean))
print(f"log-mel distance: render(short room) vs render(long room) = {np.mean(d_sl):.3f}")
print(f"                  render(short) vs clean codec = {np.mean(d_clean_s):.3f} | render(long) vs clean codec = {np.mean(d_clean_l):.3f}")