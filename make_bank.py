import numpy as np, pickle
from degrade import make_rir

rng = np.random.default_rng(1)
rts = rng.uniform(0.2, 1.0, 400)
bank = [(make_rir(rt, rng).astype(np.float32), float(rt)) for rt in rts]
pickle.dump({"train": bank[:320], "test": bank[320:]}, open("rir_bank.pkl", "wb"))
print("ok", len(bank))