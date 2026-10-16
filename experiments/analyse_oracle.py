import numpy as np, librosa, soundfile as sf

def logmel(y):
    m = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=1024, hop_length=200, n_mels=80)
    return np.log(m + 1e-6)

def floor_db(y):                                    # level of the quietest 10% of frames
    r = librosa.feature.rms(y=y, frame_length=800, hop_length=200)[0]
    return 20 * np.log10(np.percentile(r, 10) + 1e-9)

def dist(a, b):
    n = min(len(a), len(b))
    return np.abs(logmel(a[:n]) - logmel(b[:n])).mean()

clean = sf.read("o_clean.wav")[0]
cc = sf.read("o_clean_codec.wav")[0]
print(f"reference: clean floor {floor_db(clean):.1f} dB | codec error on clean {dist(cc, clean):.2f}\n")
print(f"{'case':8s} {'floor':>9s} {'(codec)':>9s} {'d(codec,degraded)':>18s} {'d(codec,clean)':>16s} {'d(degraded,clean)':>18s}")
for tag in ["reverb", "noise", "babble", "mixed"]:
    deg = sf.read(f"o_{tag}_degrade.wav")[0]
    cod = sf.read(f"o_{tag}_codec.wav")[0]
    print(f"{tag:8s} {floor_db(deg):8.1f}  {floor_db(cod):8.1f}  {dist(cod, deg):16.2f}  {dist(cod, clean):15.2f}  {dist(deg, clean):17.2f}")