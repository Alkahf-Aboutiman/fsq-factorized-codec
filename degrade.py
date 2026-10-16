import numpy as np
import pyroomacoustics as pra
from scipy.signal import fftconvolve

SR = 16000

def colored_noise(n, kind, rng):
    w = rng.standard_normal(n)
    if kind == "white":
        return w
    f = np.fft.rfftfreq(n, 1 / SR); f[0] = f[1]
    shape = {"pink": 1 / np.sqrt(f), "brown": 1 / f}[kind]
    x = np.fft.irfft(np.fft.rfft(w) * shape, n)
    return x / (x.std() + 1e-9)

def babble(n, pool, rng, k=4):
    """Somme de k extraits de parole (pool = liste de tableaux 1-D)."""
    out = np.zeros(n)
    for _ in range(k):
        y = pool[rng.integers(len(pool))]
        if len(y) < n:
            y = np.tile(y, int(np.ceil(n / len(y))))
        s = rng.integers(0, len(y) - n + 1)
        out += y[s:s + n] / (y[s:s + n].std() + 1e-9)
    return out

def make_rir(rt60, rng):
    for _ in range(20):
        dim = [rng.uniform(3, 10), rng.uniform(3, 8), rng.uniform(2.5, 4)]
        try:
            absorption, order = pra.inverse_sabine(rt60, dim)
        except ValueError:
            continue
        room = pra.ShoeBox(dim, fs=SR, materials=pra.Material(absorption), max_order=order)
        src = [rng.uniform(0.5, d - 0.5) for d in dim]
        mic = [rng.uniform(0.5, d - 0.5) for d in dim]
        if np.linalg.norm(np.array(src) - np.array(mic)) < 0.5:
            continue
        room.add_source(src); room.add_microphone(mic)
        room.compute_rir()
        rir = np.asarray(room.rir[0][0])
        return rir[np.argmax(np.abs(rir)):]        # remove the direct-path delay: keeps alignment with the clean signal
    raise RuntimeError("RIR impossible")

def degrade(clean, rng, pool, rt60=None, snr=None, noise=None, rir=None):
    rt60 = rng.uniform(0.2, 1.0) if rt60 is None else rt60
    snr = rng.uniform(0, 20) if snr is None else snr
    noise = rng.choice(["white", "pink", "brown", "babble"]) if noise is None else noise
    rir = make_rir(rt60, rng) if rir is None else rir  # use the given RIR, or generate one from rt60
    wet = fftconvolve(clean, rir)[:len(clean)]
    wet *= clean.std() / (wet.std() + 1e-9)         # keep the speech level
    n = babble(len(wet), pool, rng) if noise == "babble" else colored_noise(len(wet), noise, rng)
    n *= wet.std() / (10 ** (snr / 20) * (n.std() + 1e-9))
    out = wet + n
    out /= max(1.0, np.abs(out).max() / 0.99)       # avoid clipping
    return out.astype(np.float32), dict(rt60=float(rt60), snr=float(snr), noise=str(noise))