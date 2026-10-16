import sys
import os; sys.path.insert(0, os.environ.get("AMPHION_PATH", "./Amphion"))

import argparse, glob, json, random
import numpy as np, soundfile as sf
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from fsq_factors import FSQFactor

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)                 # config name (A, B, C, D...)
ap.add_argument("--p-bits", type=int, required=True)     # bits per frame: prosody
ap.add_argument("--c-bits", type=int, required=True)     # bits per frame: content
ap.add_argument("--r-bits", type=int, default=0)         # bits per frame: residual (0 = none)
ap.add_argument("--steps", type=int, default=5000)
args = ap.parse_args()

def levels_for_bits(bits):                               # FSQ levels whose product is 2**bits
    return [4] * (bits // 2) + ([2] if bits % 2 else [])

dev = "cuda"

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

# --- Data: 2-second crops of LibriSpeech dev-clean
torchaudio.datasets.LIBRISPEECH("./data", url="dev-clean", download=True)

class Crops(torch.utils.data.Dataset):
    def __init__(self, root, sec=2.0):
        self.files = glob.glob(f"{root}/*/*/*.flac")
        self.n = int(sec * 16000)
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        y, _ = sf.read(self.files[i], dtype="float32")
        if len(y) < self.n:
            y = np.pad(y, (0, self.n - len(y)))
        s = random.randint(0, len(y) - self.n)
        return torch.from_numpy(y[s:s + self.n])[None]

dl = torch.utils.data.DataLoader(
    Crops("./data/LibriSpeech/dev-clean"),
    batch_size=16, shuffle=True, num_workers=4, drop_last=True)

# --- FSQ factors for the chosen config
bits = [args.p_bits, args.c_bits] + ([args.r_bits] if args.r_bits > 0 else [])
facs = nn.ModuleList([FSQFactor(levels_for_bits(b)) for b in bits]).to(dev)
opt = torch.optim.AdamW(facs.parameters(), lr=1e-3)
print(f"config {args.name}: {sum(bits)} bits/frame = {80 * sum(bits) / 1000:.2f} kbps")

def save():
    torch.save(facs.state_dict(), f"fsq_{args.name}.pt")
    json.dump({"p": args.p_bits, "c": args.c_bits, "r": args.r_bits},
              open(f"fsq_{args.name}.json", "w"))

step = 0
while step < args.steps:
    for wav in dl:
        wav = wav.to(dev)
        with torch.no_grad():
            h = enc(wav)
            _, qs, _, tgt, _ = dec(h, vq=True)           # outputs of the frozen RVQ (distillation targets)

        p, ids_p = facs[0](h)
        c, _ = facs[1](h)
        outs = [p, c]
        if len(facs) == 3:
            r, _ = facs[2](h - (p + c).detach())
            outs.append(r)
        loss = sum(F.mse_loss(o, t) for o, t in zip(outs, tgt))

        opt.zero_grad(); loss.backward(); opt.step()
        step += 1

        if step % 100 == 0:
            rel = "/".join(f"{(F.mse_loss(o, t) / t.pow(2).mean()).item():.2f}"
                           for o, t in zip(outs, tgt))
            print(f"{step} loss={loss.item():.2f} rel_err={rel} "
                  f"prosody codes used: fsq={ids_p.unique().numel()} rvq={qs[0].unique().numel()}")
        if step % 1000 == 0:
            save()
        if step >= args.steps:
            break

save()