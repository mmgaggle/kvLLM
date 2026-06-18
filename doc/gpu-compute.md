# GPU compute design

This document describes the GPU side of kLLM: how transformer inference is
implemented in HIP and how it produces and consumes the content-addressed KV
cache. All data movement is currently host-mediated (ordinary HIP H2D/D2H
copies); GPU-initiated NVMe-oF fetch into VRAM is future work (see the roadmap
in the top-level README).

## KV block layout

A page is `page_tokens` consecutive positions. A block holds all layers' K then
V for those positions, contiguous and fp16:

```
block = for l in [0, L): K[l] then V[l]
K[l], V[l] : [page_tokens][kv_heads][head_dim]   (fp16, row-major)
block_bytes = 2 * L * page_tokens * kv_heads * head_dim * sizeof(fp16)
```

This is the unit a single KV-store GET returns (and, in the disaggregated
design, the unit a single fetch faults into VRAM), so the attention kernel reads
it without a gather.

## Forward, continuation, and cache

- **Full forward** computes K/V for every position and the logits for the last
  position. On a cache miss this populates the store.
- **Continuation forward** consumes cached prefix KV as attention context and
  computes only the suffix tokens `[prefix_len, n)`. The engine finds the longest
  cached prefix (the run of leading content-key hits), loads those blocks, and
  computes the rest.
- **Incremental KV cache** keeps per-layer K/V in VRAM and decodes one token at a
  time (O(n) instead of O(n²) recompute).
- **Warm-start** loads cached prefix KV directly into the incremental cache and
  skips its prefill entirely — the content-addressed compute-skip.

## Models

Two stacks are implemented:

- A small **Llama-style** decoder (RMSNorm, GQA, RoPE, SwiGLU) with a CPU
  reference, used to validate the HIP kernels.
- **GPT-2** (124M): LayerNorm, learned positions, combined QKV, GELU, tied
  embeddings. Real weights, real text.

## Tokenizer

Tokenization runs in userspace. The device speaks token IDs; the BPE tokenizer
lives in the client. A tokenizer's identity folds into the KV namespace so cached
blocks cannot cross tokenizer versions.

## Validation

- **Numerical parity** — the HIP forward matches a CPU reference of identical
  math (max absolute logit error ~1e-7).
- **Cache invariant** — generating from loaded fp16 cached KV produces the same
  greedy output as a full recompute. This is the guarantee that makes shared KV
  safe.
- **Warm-start equivalence** — loading the cached prefix and skipping its prefill
  yields the same tokens as the cold path.

## Open questions

- **Attention kernel** — currently a hand-rolled, correctness-first kernel
  (one wavefront per query). rocWMMA / Composable Kernel or a ported
  paged-attention kernel would improve throughput.
- **Numerical reproducibility** across the cached-prefix and full-recompute paths
  (reduction order) must stay within the cache-invariant tolerance.
- **Launch overhead** — per-operation kernels are launch-bound at small models
  and batch sizes; fusion and batching are needed for the compute reductions to
  show up in wall-clock throughput.
