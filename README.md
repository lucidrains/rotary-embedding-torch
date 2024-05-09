<img src="./rope.png" width="450px"></img>

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

## Inference Key-Value Cache

When dealing with key / value caches at inference, the query position needs to be offset with the `key_value_seq_length - query_seq_length`

To make this easy, use the `rotate_queries_with_cached_keys` method

```python
q = torch.randn(1, 8, 1, 64)     # only one query at a time
k = torch.randn(1, 8, 1024, 64)  # key / values with cache concatted

q, k = rotary_emb.rotate_queries_with_cached_keys(q, k)
```

You can also do this manually like so

```python
q = rotary_emb.rotate_queries_or_keys(q, offset = k.shape[-2] - q.shape[-2])
```

## Axial Rotary Embeddings

For easy use of n-dimensional axial relative positional embedding, ie. video transformers

```python
import torch

from rotary_embedding_torch import (
    RotaryEmbedding,
    apply_rotary_emb
)

pos_emb = RotaryEmbedding(
    dim = 16,
    freqs_for = 'pixel',
    max_freq = 256
)

# queries and keys for frequencies to be rotated into
# say for a video with 8 frames, and rectangular image (feature dimension comes last)

q = torch.randn(1, 8, 64, 32, 64)
k = torch.randn(1, 8, 64, 32, 64)

# get axial frequencies - (8, 64, 32, 16 * 3 = 48)
# will automatically do partial rotary

freqs = pos_emb.get_axial_freqs(8, 64, 32)

# rotate in frequencies

q = apply_rotary_emb(freqs, q)
k = apply_rotary_emb(freqs, k)
```

## Length Extrapolatable Rotary Embeddings

In <a href="https://arxiv.org/abs/2212.10554v1">this paper</a>, they were able to fix length extrapolation issue with rotary embeddings by giving it a decay similar to ALiBi. They named this technique XPos, and you can use it by setting `use_xpos = True` on initialization.

This can only be used for autoregressive transformers

```python
import torch
from rotary_embedding_torch import RotaryEmbedding

# instantiate the positional embedding in your transformer and pass to all your attention layers

rotary_emb = RotaryEmbedding(
    dim = 32,
    use_xpos = True   # set this to True to make rotary embeddings extrapolate better to sequence lengths greater than the one used at training time
)

# mock queries and keys - dimensions should end with (seq_len, feature dimension), and any number of preceding dimensions (batch, heads, etc)

q = torch.randn(1, 8, 1024, 64) # queries - (batch, heads, seq len, dimension of head)
k = torch.randn(1, 8, 1024, 64) # keys

# apply the rotations to your queries and keys after the heads have been split out, but prior to the dot product and subsequent softmax (attention)

# instead of using `rotate_queries_or_keys`, you will use `rotate_queries_and_keys`, the rest is taken care of

q, k = rotary_emb.rotate_queries_and_keys(q, k)
```

## Interpolating Sequence Positions

This MetaAI <a href="https://arxiv.org/abs//2306.15595">paper</a> proposes simply fine-tuning on interpolations of the sequence positions for extending to longer context length for pretrained models. They show this performs much better than simply fine-tuning on the same sequence positions but extended further.

You can use this by setting the `interpolate_factor` on initialization to a value greater than `1.` (ex. if pretrained model was trained on 2048, setting `interpolate_factor = 2.` would allow fine-tuning to `2048 x 2. = 4096`)

Update: someone in the community has reported that it does not work well. please email me if you see either a positive or negative result

```python
import torch
from rotary_embedding_torch import RotaryEmbedding

rotary_emb = RotaryEmbedding(
    dim = 32,
    interpolate_factor = 2.    # add this line of code to pretrained model and fine-tune for ~1000 steps, as shown in paper
)
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

```bibtex
@inproceedings{Sun2022ALT,
    title     = {A Length-Extrapolatable Transformer},
    author    = {Yutao Sun and Li Dong and Barun Patra and Shuming Ma and Shaohan Huang and Alon Benhaim and Vishrav Chaudhary and Xia Song and Furu Wei},
    year      = {2022}
}
```

```bibtex
@inproceedings{Chen2023ExtendingCW,
    title   = {Extending Context Window of Large Language Models via Positional Interpolation},
    author  = {Shouyuan Chen and Sherman Wong and Liangjian Chen and Yuandong Tian},
    year    = {2023}
}
```

```bibtex
@misc{bloc97-2023
    title   = {NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation.},
    author  = {/u/bloc97},
    url     = {https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/}
}
```
