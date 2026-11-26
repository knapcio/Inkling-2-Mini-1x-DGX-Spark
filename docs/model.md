# The model

Read from the checkpoint's `config.json` at revision `ed40e7da`:

| | |
|---|---|
| Parameters | 26.4B total, 3.4B active per pass (model card) |
| Layers | 24 stored; layers 6-17 form a looped core that runs 3 times, so a token passes 48 layer applications |
| Experts | 128 routed experts per MoE layer, top-8, no shared expert |
| Attention | 20 sliding-window layers (window 1,024) and 4 global layers (5, 11, 17, 23); 24 query heads, 8 KV heads, head dim 128, relative position bias; a short convolution (kernel 4) in every layer. Each pass of the looped core keeps its own KV and convolution state |
| Context | up to 1,048,576 tokens |
| Precision | routed experts NVFP4 in layers 0-19 and MXFP8 in layers 20-23 (`quant_method: inkling2`, ModelOpt); attention, embeddings and LM head BF16 |
| Bundled draft model | DSpark, 3 layers, block size 8 (`mtp.safetensors`, 0.64 GB); used with `SPEC=dspark` and `ENGINE=vllm` |
| Inputs | text and images (23-layer vision tower); text out. This recipe serves text only |
| Reasoning | always thinks; `reasoning_effort` none / minimal / low / medium / high / xhigh / max or 0.0-0.99 (default high = 0.9) |
| On disk | 20.2 GB |
| Licence | Apache-2.0 |

The default drafter, read from its `config.json` at revision `4f9088ea`:

| | |
|---|---|
| Drafter | DFlash2 by incoai, [incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) |
| Layers | 5 sliding-window attention layers (window 2,048), hidden size 2,304, BF16 |
| Drafting | block diffusion: a block of 8 positions per pass (7 drafts per verify step), 16 candidates per position, a selector traces one path |
| Inputs | target features from layer visits 5, 23, 41 and 48 (the last after the final norm) |
| On disk | ~1 GB |
| Licence | Apache-2.0 |
