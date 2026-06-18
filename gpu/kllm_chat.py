#!/usr/bin/env python3
# Userspace client for /dev/kllm (Granite-backed): apply the instruct chat
# template, BPE tokenize, do the token-id transaction with the device, decode.
import sys, os, glob
from tokenizers import Tokenizer

prompt = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"
n_new  = int(sys.argv[2]) if len(sys.argv) > 2 else 64
temp   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
EOS    = 100257  # Granite <|end_of_text|>

here = os.path.dirname(os.path.abspath(__file__))
tjson = os.path.join(here, "..", "models", "granite", "tokenizer.json")
if not os.path.exists(tjson):
    sys.exit("granite tokenizer.json not found (run gpu/export_granite.py first)")
tok = Tokenizer.from_file(tjson)

text = (f"<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>\n"
        f"<|start_of_role|>assistant<|end_of_role|>")
ids = tok.encode(text).ids
req = (f"{n_new} {temp} " + " ".join(map(str, ids))).encode()

fd = os.open("/dev/kllm", os.O_RDWR)          # one fd: write request, read reply
os.write(fd, req)
data = b""
while True:
    chunk = os.read(fd, 65536)
    if not chunk:
        break
    data += chunk
os.close(fd)

lines = data.decode().splitlines()
gen = [int(x) for x in lines[0].split()] if lines and lines[0].strip() else []
if EOS in gen:
    gen = gen[:gen.index(EOS)]
stats = next((l for l in lines if l.startswith("#")), "")

print("PROMPT :", prompt)
print("ANSWER :", tok.decode(gen).strip())
if stats:
    print("KV     :", stats.lstrip("# "))
