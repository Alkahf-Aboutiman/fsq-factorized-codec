import sys
import os; sys.path.insert(0, os.environ.get("AMPHION_PATH", "./Amphion"))

import pickle, numpy as np, torch, torch.nn.functional as F
import librosa, soundfile as sf, torchaudio
from huggingface_hub import hf_hub_download
from models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from env_models import EnvEncoder, Renderer
from degrade import degrade

CKPT = "env_model.pt"
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

ck = torch.load(CKPT)
E, R = EnvEncoder().to(dev), Renderer().to(dev)
E.load_state_dict(ck["E"]); R.load_state_dict(ck["R"]); E.eval(); R.eval()
bank = pickle.load(open("rir_bank.pkl", "rb"))["test"]          # rooms never seen in training

mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=1024, hop_length=256, n_mels=80).to(dev)
def mel_l1(x, y):
    n = min(x.shape[-1], y.shape[-1])
    lm = lambda w: torch.log(mel(w[..., :n].squeeze(1)) + 1e-5)
    return F.l1_loss(lm(x), lm(y)).item()

w = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float()[None, None].to(dev)

def lat(wav, full):
    h = enc(wav)
    _, ids, _, _, spk = dec(h, eval_vq=False, vq=True)
    return h, dec.vq2emb(ids, use_residual_code=full), spk

load = lambda n: librosa.load(librosa.ex(n), sr=16000)[0]
X, Y, Z = load("libri1"), load("libri2"), load("libri3")
n = min(len(X), len(Y), 6 * 16000)
X, Y = X[:n].astype(np.float64), Y[:n].astype(np.float64)
pool = [Z]
rng = np.random.default_rng(3)
pick = lambda rt: min(bank, key=lambda r: abs(r[1] - rt))        # unseen room with RT60 closest to rt

cases = {"dry_white": (0.25, 15, "white"), "mid_pink": (0.55, 8, "pink"), "reverb_babble": (0.9, 10, "babble")}
print(f"{'case':14s} {'room RT60':>9s} | distance to the TRUE degraded Y (log-mel, lower is better)")
with torch.no_grad():
    for tag, (rt, snr, noise) in cases.items():
        rir, rt60 = pick(rt)
        dX, _ = degrade(X, rng, pool, rt60=rt60, snr=snr, noise=noise, rir=rir)
        dY, _ = degrade(Y, rng, pool, rt60=rt60, snr=snr, noise=noise, rir=rir)   # ground truth
        h_X, _, _ = lat(w(dX), False)
        e, _ = E(h_X)                                                           # environment of X only
        _, emb_c, spk = lat(w(Y), False)
        _, emb_cf, _ = lat(w(Y), True)
        _, emb_d, spk_d = lat(w(dY), True)
        render = dec.inference(R(emb_c, e), spk)                                # clean Y + e(X)
        clean = dec.inference(emb_cf, spk)                                      # no environment
        oracle = dec.inference(emb_d, spk_d)                                    # codec on the true degraded Y
        wdY = w(dY)
        print(f"{tag:14s} {rt60:9.2f} | render {mel_l1(render, wdY):.3f} | no env {mel_l1(clean, wdY):.3f} "
              f"| codec ceiling {mel_l1(oracle, wdY):.3f}")
        for name, y in (("X_degraded", dX), ("Y_clean", Y), ("Y_true_degraded", dY),
                        ("Y_render_with_e_of_X", render[0, 0].cpu().numpy())):
            sf.write(f"d_{tag}_{name}.wav", np.asarray(y, dtype=np.float32), 16000)