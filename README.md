## Rotary Embeddings - Pytorch (wip)

A standalone library for adding <a href="https://arxiv.org/abs/2104.09864">rotary embeddings</a> to transformers, in Pytorch. Specifically it will make rotating information into any axis of a tensor easy and efficient, whether they be fixed positional or learned. My gut tells me there is something <a href="https://www.nature.com/articles/s41593-021-00821-9">more</a> to rotations that can be exploited in artificial neural networks.

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
