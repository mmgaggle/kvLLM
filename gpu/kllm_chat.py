#!/usr/bin/env python3
# Userspace client for /dev/kllm: BPE tokenize -> token-id transaction with the
# device -> detokenize. The device speaks tokens; the tokenizer lives here.
import sys, os, glob
from tokenizers import Tokenizer

prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
n_new  = int(sys.argv[2]) if len(sys.argv) > 2 else 30
temp   = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0   # 0 = greedy

tjson = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--gpt2/snapshots/*/tokenizer.json"))
if not tjson:
    sys.exit("gpt2 tokenizer.json not found")
tok = Tokenizer.from_file(tjson[0])

ids = tok.encode(prompt).ids
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
gen_ids = [int(x) for x in lines[0].split()] if lines and lines[0].strip() else []
stats = next((l for l in lines if l.startswith("#")), "")

print("PROMPT :", prompt)
print("OUTPUT :", tok.decode(ids + gen_ids))
if stats:
    print("KV     :", stats.lstrip("# "))
