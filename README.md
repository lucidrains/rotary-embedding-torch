## Rotary Embeddings - Pytorch

A standalone library for adding <a href="https://arxiv.org/abs/2104.09864">rotary embeddings</a> to transformers in Pytorch, following its success as <a href="https://blog.eleuther.ai/rotary-embeddings/">relative positional encoding</a>. Specifically it will make rotating information into any axis of a tensor easy and efficient, whether they be fixed positional or learned. This library will give you state of the art results for positional embedding, at little costs.

My gut also tells me there is something <a href="https://www.nature.com/articles/s41593-021-00821-9">more</a> to rotations that can be exploited in artificial neural networks.

## Install

```bash
$ pip install rotary-embedding-torch
```

## Usage

```python
import torch
from rotary_embedding_torch import RotaryEmbedding

# instantiate the positional embedding in your transformer and pass to all your attention layers

rotary_emb = RotaryEmbedding(dim = 32)

# mock queries and keys - dimensions should end with (seq_len, feature dimension), and any number of preceding dimensions (batch, heads, etc)

q = torch.randn(1, 8, 1024, 64) # queries - (batch, heads, seq len, dimension of head)
k = torch.randn(1, 8, 1024, 64) # keys

# apply the rotations to your queries and keys after the heads have been split out, but prior to the dot product and subsequent softmax (attention)

q = rotary_emb.rotate_queries_or_keys(q)
k = rotary_emb.rotate_queries_or_keys(k)

# then do your attention with your queries (q) and keys (k) as usual
```

If you do all the steps above correctly, you should see a dramatic improvement during training

## Axial Rotary Embeddings

For easy use of 2d axial relative positional embedding, ie. vision transformers

```python
import torch
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding, broadcat

pos_emb = RotaryEmbedding(
    dim = 32,
    freqs_for = 'pixel',
    max_freq = 256
)

# queries and keys for frequencies to be rotated into

q = torch.randn(1, 256, 256, 64)
k = torch.randn(1, 256, 256, 64)

# get frequencies for each axial
# -1 to 1 has been shown to be a good choice for images and audio

freqs_h = pos_emb(torch.linspace(-1, 1, steps = 256), cache_key = 256)
freqs_w = pos_emb(torch.linspace(-1, 1, steps = 256), cache_key = 256)

# concat the frequencies along each axial
# broadcat function makes this easy without a bunch of expands

freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

# rotate in frequencies

q = apply_rotary_emb(freqs, q)
k = apply_rotary_emb(freqs, k)
```

## Learned Rotations

For injecting learned rotations into a network. Experiments pending

Update: doesn't seem to do anything -_-, will keep trying...

```python
import torch
from torch import nn
from rotary_embedding_torch import apply_learned_rotations

x = torch.randn(1, 1024, 512)

# you can only rotate in (dim // 2) values
# ex. for 512, you can only rotate in 256 values

# say you have two sets of learned rotations of 128 values each

rots1 = nn.Linear(512, 128)(x)
rots2 = nn.Linear(512, 128)(x)

# you rotate in 256 (128 x 2) at first

x = apply_learned_rotations(rots1, x, start_index = 0)

# then you start at index 256 and rotate in the last (128 x 2)

x = apply_learned_rotations(rots2, x, start_index = 256)

# you could also concat the rotations together and pass it in all at once

rots = torch.cat((rots1, rots2), dim = -1)

x = apply_learned_rotations(rots, x)
```

## Citations

```bibtex
@misc{su2021roformer,
    title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding}, 
    author  = {Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu},
    year    = {2021},
    eprint  = {2104.09864},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```
