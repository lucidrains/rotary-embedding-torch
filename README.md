## Rotary Embeddings - Pytorch

A standalone library for adding <a href="https://arxiv.org/abs/2104.09864">rotary embeddings</a> to transformers in Pytorch, following its success as <a href="https://blog.eleuther.ai/rotary-embeddings/">relative positional encoding</a>. Specifically it will make rotating information into any axis of a tensor easy and efficient, whether they be fixed positional or learned. My gut tells me there is something <a href="https://www.nature.com/articles/s41593-021-00821-9">more</a> to rotations that can be exploited in artificial neural networks.

## Install

```bash
$ pip install rotary-embedding-torch
```

## Usage

```python
import torch
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding

# instantiate the positional embedding in your transformer and pass to all your attention layers

pos_emb = RotaryEmbedding(dim = 32)

# generate the rotations

freqs = pos_emb(torch.arange(1024), cache_key = 1024) # cache with a key that is the sequence length, so that it does not need to recompute

# mock queries and keys

q = torch.randn(1, 1024, 64) # queries - (batch, seq len, dimension of head)
k = torch.randn(1, 1024, 64) # keys

# apply the rotations to your queries and keys after the heads have been split out, but prior to the dot product and subsequent softmax (attention)

freqs = freqs[None, ...] # unsqueeze for batch dimension
q = apply_rotary_emb(freqs, q)
k = apply_rotary_emb(freqs, k)

# then do your attention with your queries (q) and keys (k)
```

If you do all the steps above correctly, you should see a dramatic improvement during training

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
