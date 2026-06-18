#!/usr/bin/env python3
# Export IBM Granite 4.0 (dense transformer) safetensors to a flat fp32 blob.
# HF Linear weights are [out, in]; we transpose to [in, out] to match the device
# matmul convention. Header carries dims and Granite's scalar multipliers.
#
# safetensors is parsed directly so bf16 can be widened to fp32 without torch.
import sys, struct, json
import numpy as np

src  = sys.argv[1] if len(sys.argv) > 1 else "models/granite/model.safetensors"
cfgp = sys.argv[2] if len(sys.argv) > 2 else "models/granite/config.json"
out  = sys.argv[3] if len(sys.argv) > 3 else "models/granite/granite.weights"

# ---- minimal safetensors reader (bf16/f16/f32 -> fp32) ----
_fh = open(src, "rb")
_hlen = struct.unpack("<Q", _fh.read(8))[0]
_hdr = json.loads(_fh.read(_hlen))
_blob = 8 + _hlen
def tensor(name):
    m = _hdr[name]; s, e = m["data_offsets"]
    _fh.seek(_blob + s)
    raw = _fh.read(e - s)
    dt = m["dtype"]
    if dt == "BF16":
        u = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
        a = u.view(np.float32)
    elif dt == "F16":
        a = np.frombuffer(raw, np.float16).astype(np.float32)
    elif dt == "F32":
        a = np.frombuffer(raw, np.float32)
    else:
        raise ValueError("dtype " + dt)
    return a.reshape(m["shape"])

c = json.load(open(cfgp))
L  = c["num_hidden_layers"]; H = c["hidden_size"]
NH = c["num_attention_heads"]; NKV = c["num_key_value_heads"]
HD = c.get("head_dim", H // NH); I = c["intermediate_size"]; V = c["vocab_size"]
MAGIC = 0x4752414E  # 'GRAN'

def t(k):  return np.ascontiguousarray(tensor(k), dtype=np.float32)        # as-is
def tT(k): return np.ascontiguousarray(tensor(k).T, dtype=np.float32)      # [out,in]->[in,out]

with open(out, "wb") as o:
    o.write(struct.pack("<8i", MAGIC, L, H, NH, NKV, HD, I, V))
    o.write(struct.pack("<6f", c["rope_theta"], c["rms_norm_eps"],
                        c["attention_multiplier"], c["embedding_multiplier"],
                        c["residual_multiplier"], c["logits_scaling"]))
    def w(a): o.write(a.tobytes())
    w(t("model.embed_tokens.weight"))      # [V][H]  (also tied lm_head)
    w(t("model.norm.weight"))              # [H]
    for i in range(L):
        p = f"model.layers.{i}."
        w(t(p + "input_layernorm.weight"))
        w(tT(p + "self_attn.q_proj.weight"))          # [H][H]
        w(tT(p + "self_attn.k_proj.weight"))          # [H][KV]
        w(tT(p + "self_attn.v_proj.weight"))          # [H][KV]
        w(tT(p + "self_attn.o_proj.weight"))          # [H][H]
        w(t(p + "post_attention_layernorm.weight"))
        w(tT(p + "shared_mlp.input_linear.weight"))   # [H][2I]  (gate|up)
        w(tT(p + "shared_mlp.output_linear.weight"))  # [I][H]

print(f"wrote {out}: L={L} H={H} NH={NH} NKV={NKV} HD={HD} I={I} V={V}")
