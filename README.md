# kLLM

An experimental LLM serving engine built around a single idea: **KV cache and
model weights are immutable, content-addressed blocks, and the GPU faults them
into VRAM by hash.** Prefix caching becomes demand paging; the backing store can
live anywhere from local memory to a disaggregated NVMe/RADOS namespace.

The serving interface is a character device (`/dev/kllm`), so clients interact
with the engine through ordinary file operations rather than an HTTP API.

> Status: research prototype. The compute path, the content-addressed KV cache,
> and the character-device interface are implemented and tested on a single host
> (AMD GPU, ROCm). The disaggregated off-box store (RADOS + GPU-initiated
> NVMe-oF fetch) is the next phase and not yet implemented — see *Roadmap*.

## Concept

A KV-cache block is identified by a Merkle hash over its token prefix:

```
block_key[0] = H(namespace_seed ‖ tokens_of_block_0)
block_key[i] = H(block_key[i-1] ‖ tokens_of_block_i)
```

Identical prefixes therefore chain to identical keys and share the same physical
block — within a request, across requests, and (in the disaggregated design)
across nodes. A request that shares a prefix with a prior one loads that prefix's
KV from the store and skips recomputing it.

```
client ──▶ tokenizer (userspace) ──▶ /dev/kllm ──▶ paged-attention engine
                                                        │
                        per page: block_key = H(parent_key ‖ tokens)
                                                        │
                   ┌─────────────────────────────────────┴──────────────┐
                cache hit                                            cache miss
       load the KV block from the                         compute KV on the GPU,
       content store; skip its compute                    store under its key
```

## Implemented

All of the following are built and have runnable checks (single host, AMD
gfx1151 / ROCm):

- **`/dev/kllm`** — a CUSE character device. A request is a single descriptor
  transaction: write the prompt, read the completion.
- **Transformer inference in HIP**, written from scratch (no PyTorch or
  llama.cpp). A Llama-style stack (RMSNorm, GQA, RoPE, SwiGLU) is validated
  against a CPU reference to a max absolute logit error of 1.2e-7. A GPT-2 stack
  (LayerNorm, learned positions, GELU, tied embeddings) loads the real 124M
  weights and generates coherent text.
- **Content-addressed paged KV cache** — fp16 KV blocks keyed by prefix Merkle
  hash, in a store that mirrors the SPDK `kvdev` interface so a real
  disaggregated backend can replace it without touching the engine.
- **Cache-correctness invariant** — generating from loaded fp16 cached KV
  produces the same greedy output as a full recompute (verified token-for-token,
  not assumed). This is the property that makes shared KV safe.
- **Incremental KV cache and warm-start** — O(n) decode instead of O(n²)
  recompute, and a cache hit loads the prefix KV and skips its prefill entirely.
- **Live serving** — the daemon detects cached prompt pages, retrieves them from
  the store, and warm-starts generation. Supports greedy and temperature
  sampling.

| Component | Check |
|---|---|
| Char-device spine + content addressing (CPU) | `make test` |
| HIP forward vs CPU reference | `make gpu-test` |
| Greedy decode + content-addressed KV store | `make gpu-decode` |
| Cache invariant (cached prefix == full recompute) | `make gpu-cache-test` |
| Incremental KV cache == recompute | `make gpu-kvcache` |
| Warm-start (load prefix, skip prefill) | `make gpu-warmstart` |
| Real GPT-2 generation | `make gpt2` |
| `/dev/kllm` serving GPT-2 with warm-start | `make serve` |

## Build and run

The CPU spine builds and tests without a GPU:

```sh
make test
```

The GPU components require ROCm / `hipcc`:

```sh
make gpu-test gpu-decode gpu-cache-test gpu-kvcache gpu-warmstart
```

Real GPT-2 generation (weights are fetched from the local Hugging Face cache or
downloaded on first export):

```sh
python3 gpu/export_gpt2.py            # one-time: safetensors -> flat fp32 weights
make gpt2
python3 gpu/run_gpt2.py "The meaning of life is" 30
```

Serving through `/dev/kllm` (the device node requires the `cuse` kernel module):

```sh
sudo dnf install kernel-modules-extra-$(uname -r)   # Fedora: provides cuse.ko
sudo modprobe cuse
make serve && sudo ./gpu/kllm-serve -f &
sudo chmod 666 /dev/kllm
python3 gpu/kllm_chat.py "Once upon a time in a land far away" 25        # greedy
python3 gpu/kllm_chat.py "Once upon a time in a land far away" 25 0.8    # sampling
```

Sending the same prompt twice: the second request reports `prefill_skipped N` —
its prompt KV is loaded from the content store rather than recomputed.

## Design notes

- **Namespace encodes compute context; the key is a pure prefix hash.** One KV
  namespace per `(model, weights version, dtype, quantization, attention
  config)`. Blocks from different runtimes cannot collide, so a hit can only be
  served from a matching configuration.
- **The tokenizer runs in userspace.** Tokenization is not the bottleneck and is
  poorly suited to the original in-kernel/eBPF idea (data-dependent BPE merge
  loops). The device speaks token IDs; the BPE tokenizer lives in the client.
- **fp16 KV is validated, not assumed.** The cache-invariant test confirms that
  fp16 quantization of cached KV does not change greedy output.

See [doc/gpu-compute.md](doc/gpu-compute.md) for the GPU compute design.

## Layout

```
src/            CPU spine: char device, tokenizer, content hashing, KV store
gpu/kmodel_*    Llama-style HIP forward and the correctness harnesses
gpu/gpt2.*      GPT-2 in HIP: forward, incremental KV cache, warm-start
gpu/kllm_serve  the /dev/kllm serving daemon
gpu/*.py        userspace tokenizer front-ends (export, generate, chat)
```

## Roadmap

- **Disaggregated store** — replace the in-memory KV store with the SPDK
  `rados-nkv` backend (RADOS-backed NVMe key-value), widen keys to 256 bits, and
  fetch KV blocks and weights directly into VRAM via GPU-initiated NVMe-oF. This
  is the design's original goal and the point at which the cache becomes
  fleet-wide rather than single-host.
- **eBPF I/O fast path** — transport demultiplexing and hash-based steering on
  the data plane.
- **Performance** — the current kernels are correctness-first and per-operation;
  at small models and batch sizes they are launch-bound. Kernel fusion and
  batching are needed before the algorithmic wins (e.g. the KV cache's 68×
  reduction in token-forwards) translate to wall-clock throughput.
