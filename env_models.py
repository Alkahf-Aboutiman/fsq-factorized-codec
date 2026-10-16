import torch, torch.nn as nn, torch.nn.functional as F

NOISES = ["white", "pink", "brown", "babble"]

class EnvEncoder(nn.Module):
    def __init__(self, d=256, de=128):
        super().__init__()
        self.inp = nn.Conv1d(256, d, 3, padding=1)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 512, 0.1, batch_first=True), 2)
        self.out = nn.Linear(d, de)
        self.heads = nn.Linear(de, 2 + len(NOISES))

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