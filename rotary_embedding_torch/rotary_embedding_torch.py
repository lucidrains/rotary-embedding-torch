import torch
from torch import nn, einsum

from einops import rearrange

def exists(val):
    return val is not None

def rotate_half(x):
    x = rearrange(x, 'b n (d r) -> b n d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_emb(freqs, t, start_index = 0):
    rot_dim = freqs.shape[-1]
    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
    t_left, t, t_right = t[..., :start_index], t[..., start_index:rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t_left, t, t_right), dim = -1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        freqs = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.cache = dict()
        self.register_buffer('freqs', freqs)

    def forward(self, t, cache_key = None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        freqs = self.freqs

        freqs = torch.einsum('i, j -> i j', t.type(freqs.dtype), freqs)
        freqs = torch.stack((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, '... n r -> ... (n r)')

        if exists(cache_key):
            self.cache[cache_key] = freqs

        return freqs
