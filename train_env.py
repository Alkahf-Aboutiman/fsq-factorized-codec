import sys
import os; sys.path.insert(0, os.environ.get("AMPHION_PATH", "./Amphion"))

import argparse, glob, pickle, random
import numpy as np, soundfile as sf
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from degrade import degrade          # degrade() must accept the `rir` argument

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--batch", type=int, default=8)      # lower to 4 if you get "out of memory"
args = ap.parse_args()
dev = "cuda"
NOISES = ["white", "pink", "brown", "babble"]

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
for m in (enc, dec):
    m.eval().to(dev)
    for p_ in m.parameters():
        p_.requires_grad_(False)

# --- Data: pairs (a, b) degraded with the SAME room / SNR / noise type
bank = pickle.load(open("rir_bank.pkl", "rb"))   # {"train": [(rir, rt60)], "test": [...]}

class PairDS(torch.utils.data.Dataset):
    def __init__(self, root, rirs, sec=2.0, n_pool=60):
        self.files = sorted(glob.glob(f"{root}/*/*/*.flac"))
        self.rirs, self.n, self.n_pool, self.pool = rirs, int(sec * 16000), n_pool, None

    def __len__(self):
        return len(self.files)

    def crop(self, path):
        y, _ = sf.read(path, dtype="float32")
        if len(y) < self.n:
            y = np.pad(y, (0, self.n - len(y)))
        s = random.randint(0, len(y) - self.n)
        return y[s:s + self.n].astype(np.float64)

    def __getitem__(self, i):
        rng = np.random.default_rng()
        if self.pool is None:        # loaded once per worker (used for babble noise)
            self.pool = [sf.read(f, dtype="float32")[0] for f in random.sample(self.files, self.n_pool)]
        a, b = self.crop(self.files[i]), self.crop(random.choice(self.files))
        rir, rt60 = self.rirs[rng.integers(len(self.rirs))]
        snr, noise = rng.uniform(0, 20), NOISES[rng.integers(len(NOISES))]
        da, _ = degrade(a, rng, self.pool, rt60=rt60, snr=snr, noise=noise, rir=rir)
        db, _ = degrade(b, rng, self.pool, rt60=rt60, snr=snr, noise=noise, rir=rir)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float()[None]
        return t(a), t(da), t(db), torch.tensor([rt60, snr, NOISES.index(noise)], dtype=torch.float32)

def loader(root, rirs, shuffle):
    return torch.utils.data.DataLoader(PairDS(root, rirs), batch_size=args.batch, shuffle=shuffle,
                                       num_workers=6, drop_last=True)

dl = loader("./data/LibriSpeech/dev-clean", bank["train"], True)
dl_val = loader("./data/LibriSpeech/test-clean", bank["test"], False)   # rooms never seen in training

# --- Models
class EnvEncoder(nn.Module):
    """Degraded latents [B,256,T] -> environment vector e [B,128] (+ auxiliary predictions)."""
    def __init__(self, d=256, de=128):
        super().__init__()
        self.inp = nn.Conv1d(256, d, 3, padding=1)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 512, 0.1, batch_first=True), 2)
        self.out = nn.Linear(d, de)
        self.heads = nn.Linear(de, 2 + len(NOISES))          # RT60, SNR, noise type

    def forward(self, h):
        e = self.out(self.tf(self.inp(h).transpose(1, 2)).mean(1))
        return e, self.heads(e)

class Block(nn.Module):
    def __init__(self, c, de, dil):
        super().__init__()
        self.norm = nn.GroupNorm(1, c)
        self.film = nn.Linear(de, 2 * c)
        self.conv = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil)
        self.pw = nn.Conv1d(c, c, 1)

    def forward(self, x, e):
        g, b = self.film(e).unsqueeze(-1).chunk(2, 1)
        return x + self.pw(F.gelu(self.conv(self.norm(x) * (1 + g) + b)))

class Renderer(nn.Module):
    """Clean latents (prosody + content) + e -> degraded latents. Starts as the identity."""
    def __init__(self, c=256, de=128):
        super().__init__()
        self.inp = nn.Conv1d(256, c, 1)
        self.blocks = nn.ModuleList([Block(c, de, d) for d in (1, 2, 4, 8, 16, 1)])
        self.out = nn.Conv1d(c, 256, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, emb_c, e):
        x = self.inp(emb_c)
        for b in self.blocks:
            x = b(x, e)
        return emb_c + self.out(x)

E, R = EnvEncoder().to(dev), Renderer().to(dev)
opt = torch.optim.AdamW(list(E.parameters()) + list(R.parameters()), lr=3e-4)

# --- Helpers
mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=1024, hop_length=256, n_mels=80).to(dev)

def mel_l1(x, y):
    """L1 distance between log-mel spectrograms of two waveforms [B,1,T]."""
    n = min(x.shape[-1], y.shape[-1])
    lm = lambda w: torch.log(mel(w[..., :n].squeeze(1)) + 1e-5)
    return F.l1_loss(lm(x), lm(y))

@torch.no_grad()
def latents(wav, full):
    """Quantized FACodec embedding (+ timbre vector); full=True includes the residual codes."""
    h = enc(wav)
    _, ids, _, _, spk = dec(h, eval_vq=False, vq=True)
    return h, dec.vq2emb(ids, use_residual_code=full), spk

W = 48    # latent frames decoded per example for the mel loss (0.6 s at 80 frames/s)

def losses(a, da, db, lab):
    with torch.no_grad():
        _, emb_c, spk_a = latents(a, False)    # clean a, no residual: input of R
        _, emb_a, _ = latents(da, True)        # degraded a, full: latent target
        h_b, _, _ = latents(db, False)         # degraded b (other utterance, same room): only input of E
    e, aux = E(h_b)
    pred = R(emb_c, e)
    rel = F.mse_loss(pred, emb_a) / emb_a.pow(2).mean()
    # Decode only a random window: the decoder is convolutional, and this cuts memory ~3x
    s = random.randint(0, pred.shape[-1] - W)
    wav_hat = dec.inference(pred[..., s:s + W], spk_a)       # gradient flows through the frozen decoder
    mel_loss = mel_l1(wav_hat, da[..., s * 200:(s + W) * 200])   # 200 samples per latent frame
    rt, snr, typ = lab[:, 0], lab[:, 1], lab[:, 2].long()
    aux_loss = (F.mse_loss(aux[:, 0], (rt - 0.6) / 0.3) + F.mse_loss(aux[:, 1], (snr - 10) / 6)
                + F.cross_entropy(aux[:, 2:], typ))
    return rel + mel_loss + 0.1 * aux_loss, rel, mel_loss, aux

@torch.no_grad()
def validate(n_batches=10):
    E.eval(); R.eval()
    m = {k: [] for k in ("mel", "mel_mismatch", "mel_no_env", "rt", "acc")}
    for i, (a, da, db, lab) in enumerate(dl_val):
        if i >= n_batches:
            break
        a, da, db, lab = a.to(dev), da.to(dev), db.to(dev), lab.to(dev)
        _, emb_c, spk_a = latents(a, False)
        _, emb_cf, _ = latents(a, True)        # clean latents WITH residual
        h_b, _, _ = latents(db, False)
        e, aux = E(h_b)
        m["mel"].append(mel_l1(dec.inference(R(emb_c, e), spk_a), da).item())
        m["mel_mismatch"].append(mel_l1(dec.inference(R(emb_c, e.roll(1, 0)), spk_a), da).item())  # e from another room
        m["mel_no_env"].append(mel_l1(dec.inference(emb_cf, spk_a), da).item())   # decode the clean latents: no environment
        m["rt"].append(((aux[:, 0] * 0.3 + 0.6 - lab[:, 0]) ** 2).mean().item())
        m["acc"].append((aux[:, 2:].argmax(1) == lab[:, 2].long()).float().mean().item())
    E.train(); R.train()
    return {k: float(np.sqrt(np.mean(v)) if k == "rt" else np.mean(v)) for k, v in m.items()}

# --- Training
step = 0
while step < args.steps:
    for a, da, db, lab in dl:
        a, da, db, lab = a.to(dev), da.to(dev), db.to(dev), lab.to(dev)
        loss, rel, mel_l, _ = losses(a, da, db, lab)
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step % 100 == 0:
            print(f"{step} loss={loss.item():.3f} rel={rel.item():.4f} mel={mel_l.item():.3f}")
        if step % 1000 == 0 or step == args.steps:
            v = validate()
            print(f"  VAL unseen rooms: mel={v['mel']:.3f} | e from another room={v['mel_mismatch']:.3f} | "
                  f"no environment={v['mel_no_env']:.3f} | RT60 RMSE={v['rt']:.3f} s | noise-type acc={v['acc']:.2f}")
            torch.save({"E": E.state_dict(), "R": R.state_dict()}, "env_model.pt")
        if step >= args.steps:
            break