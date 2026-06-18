# gpu/ — HIP compute

The GPU side of kLLM: transformer inference in HIP (ROCm), plus the
content-addressed KV cache and warm-start logic. Not built by the default
`make` (which stays CPU-only); build the targets explicitly with `hipcc`
available. See [../doc/gpu-compute.md](../doc/gpu-compute.md) for the design.

Contents:

- `kmodel.{h,cpp,hip}` — a small Llama-style decoder and a CPU reference, used to
  validate the HIP kernels (forward parity, KV block layout, cache invariant).
- `gpt2.{h,hip}` — GPT-2 (124M) in HIP: forward pass, incremental KV cache,
  warm-start from cached prefix KV.
- `kllm_serve.cpp` — the `/dev/kllm` serving daemon (GPT-2 + content-addressed
  KV store).
- `*_test.cpp` — correctness/timing harnesses (`make gpu-test gpu-decode
  gpu-cache-test gpu-kvcache gpu-warmstart`).
- `*.py` — userspace tokenizer front-ends: `export_gpt2.py` (weights),
  `run_gpt2.py` (generate), `kllm_chat.py` (talk to `/dev/kllm`).

Build notes:

- Device kernels live in `.hip` files (hipcc); host glue is plain `.cpp` (g++).
  Keeping host code out of hipcc avoids device-side resolution of host symbols.
- The C core is compiled `-fPIC` (hipcc links a PIE); the link adds `-lamdhip64`.
- kLLM C headers carry `extern "C"` guards so the C++/HIP code links against
  the C core.
