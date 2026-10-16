import sys
import os; sys.path.insert(0, os.environ.get("AMPHION_PATH", "./Amphion"))

import torch, librosa, soundfile as sf
from huggingface_hub import hf_hub_download
from models.codec.ns3_codec import FACodecEncoder, FACodecDecoder

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
enc.eval(); dec.eval()

def load(name):
    y, _ = librosa.load(librosa.ex(name), sr=16000)
    return torch.from_numpy(y).float()[None, None, :]

@torch.no_grad()
def encode(wav):
    h = enc(wav)
    if hasattr(enc, "get_prosody_feature"):
        p = enc.get_prosody_feature(wav)
        _, ids, _, _, spk = dec(h, p, eval_vq=False, vq=True)
    else:
        _, ids, _, _, spk = dec(h, eval_vq=False, vq=True)
    return ids, spk   # ids: [prosody(1), content(2), residual(3), B, T]

@torch.no_grad()
def decode(ids, spk, use_residual=True):
    emb = dec.vq2emb(ids, use_residual_code=use_residual)
    return dec.inference(emb, spk)[0, 0].cpu().numpy()

ids_a, spk_a = encode(load("libri1"))
ids_b, spk_b = encode(load("libri2"))

# 1) Reconstruction of A (reference)
sf.write("a_recon.wav", decode(ids_a, spk_a), 16000)

# 2) Contenu + prosodie de A, timbre de B
sf.write("a_timbre_b.wav", decode(ids_a, spk_b, use_residual=False), 16000)

# 3) Contenu de A, prosodie de B, timbre de A
T = min(ids_a.shape[-1], ids_b.shape[-1])
mix = torch.cat([ids_b[:1, :, :T], ids_a[1:, :, :T]], dim=0)
sf.write("a_prosody_b.wav", decode(mix, spk_a), 16000)
print("ok")