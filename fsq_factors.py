import torch, torch.nn as nn, torch.nn.functional as F

def round_ste(z):
    return z + (z.round() - z).detach()

class FSQ(nn.Module):
    def __init__(self, levels):
        super().__init__()
        self.register_buffer("L", torch.tensor(levels), persistent=False)
        self.register_buffer("basis", torch.cumprod(torch.tensor([1] + levels[:-1]), 0), persistent=False)

    def bound(self, z):
        half_l = (self.L - 1) * (1 + 1e-3) / 2
        offset = torch.where(self.L % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):                  # z: [B, T, d] -> codes in [-1, 1]
        return round_ste(self.bound(z)) / (self.L // 2)

    def indices(self, codes):               # codes -> entier unique par frame
        hw = self.L // 2
        return ((codes * hw + hw).round().long() * self.basis).sum(-1)

class FSQFactor(nn.Module):
    """Remplace un ResidualVQ : [B,256,T] -> [B,256,T] + indices [B,T]."""
    def __init__(self, levels, dim=256):
        super().__init__()
        self.fsq = FSQ(levels)
        self.pin = nn.Conv1d(dim, len(levels), 1)
        self.pout = nn.Conv1d(len(levels), dim, 1)

    def forward(self, x):
        codes = self.fsq.quantize(self.pin(x).transpose(1, 2))
        return self.pout(codes.transpose(1, 2)), self.fsq.indices(codes)