#!/usr/bin/env python3
# Userspace front-end for the GPT-2 HIP model: real BPE tokenize -> generate
# (on GPU via gpu/gpt2_gen) -> detokenize.
import sys, os, glob, subprocess
from tokenizers import Tokenizer

prompt = sys.argv[1] if len(sys.argv) > 1 else "The meaning of life is"
n_new  = sys.argv[2] if len(sys.argv) > 2 else "30"

tjson = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--gpt2/snapshots/*/tokenizer.json"))
if not tjson:
    sys.exit("gpt2 tokenizer.json not found in HF cache")
tok = Tokenizer.from_file(tjson[0])

ids = tok.encode(prompt).ids
here = os.path.dirname(os.path.abspath(__file__))
binp = os.path.join(here, "gpt2_gen")
wts  = os.path.join(here, "..", "models", "gpt2", "gpt2.weights")

r = subprocess.run([binp, wts, str(n_new)],
                   input=" ".join(map(str, ids)),
                   capture_output=True, text=True)
sys.stderr.write(r.stderr)
if r.returncode != 0:
    sys.exit(r.returncode)

gen = [int(x) for x in r.stdout.split()]
print("PROMPT :", prompt)
print("CONT   :", tok.decode(gen))
print("FULL   :", tok.decode(ids + gen))
