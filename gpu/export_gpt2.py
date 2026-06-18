#!/usr/bin/env python3
# Export HF GPT-2 safetensors to a flat fp32 blob the HIP loader reads.
# Layout: header (6x int32) then tensors in a fixed order, all little-endian fp32.
import sys, struct
import numpy as np
from safetensors import safe_open

src = sys.argv[1] if len(sys.argv) > 1 else "models/gpt2/model.safetensors"
out = sys.argv[2] if len(sys.argv) > 2 else "models/gpt2/gpt2.weights"

f = safe_open(src, framework="numpy")
def g(k): return np.ascontiguousarray(f.get_tensor(k), dtype=np.float32)

E, L, H, P, V = 768, 12, 12, 1024, 50257
MAGIC = 0x47505432  # 'GPT2'

with open(out, "wb") as o:
    o.write(struct.pack("<6i", MAGIC, L, E, H, P, V))
    def w(a): o.write(a.tobytes())
    w(g("wte.weight")); w(g("wpe.weight"))
    w(g("ln_f.weight")); w(g("ln_f.bias"))
    for i in range(L):
        p = f"h.{i}."
        for name in ("ln_1.weight", "ln_1.bias",
                     "attn.c_attn.weight", "attn.c_attn.bias",
                     "attn.c_proj.weight", "attn.c_proj.bias",
                     "ln_2.weight", "ln_2.bias",
                     "mlp.c_fc.weight", "mlp.c_fc.bias",
                     "mlp.c_proj.weight", "mlp.c_proj.bias"):
            w(g(p + name))

print(f"wrote {out}: L={L} E={E} H={H} P={P} V={V}")
