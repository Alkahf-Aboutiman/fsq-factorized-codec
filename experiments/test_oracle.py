import sys
import os; sys.path.insert(0, os.environ.get("AMPHION_PATH", "./Amphion"))

import numpy as np, torch, librosa, soundfile as sf
from huggingface_hub import hf_hub_download
from models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from degrade import degrade

dev = "cuda"
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

@torch.no_grad()
def roundtrip(y):                                   # full encode + decode (with residual)
    w = torch.from_numpy(y).float()[None, None].to(dev)
    _, ids, _, _, spk = dec(enc(w), eval_vq=False, vq=True)
    return dec.inference(dec.vq2emb(ids, use_residual_code=True), spk)[0, 0].cpu().numpy()

load = lambda n: librosa.load(librosa.ex(n), sr=16000)[0]
clean = load("libri1")[:16000 * 6]
pool = [load("libri2"), load("libri3")]
rng = np.random.default_rng(0)

sf.write("o_clean.wav", clean, 16000)
for tag, (rt60, snr, noise) in {"reverb": (0.8, 40, "white"),
                                "noise": (0.2, 5, "pink"),
                                "babble": (0.3, 5, "babble"),
                                "mixed": (0.6, 10, "babble")}.items():
    deg, lab = degrade(clean.astype(np.float64), rng, pool, rt60=rt60, snr=snr, noise=noise)
    sf.write(f"o_{tag}_degrade.wav", deg, 16000)
    sf.write(f"o_{tag}_codec.wav", roundtrip(deg), 16000)
    print(tag, lab)
    
sf.write("o_clean_codec.wav", roundtrip(clean), 16000)