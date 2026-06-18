#!/usr/bin/env python3
# Userspace front-end for the Granite HIP model: tokenize -> generate -> detokenize.
# With --chat, wraps the prompt in Granite's instruct chat template.
import sys, os, subprocess
from tokenizers import Tokenizer

args = [a for a in sys.argv[1:] if a != "--chat"]
chat = "--chat" in sys.argv
prompt = args[0] if len(args) > 0 else "The capital of France is"
n_new  = args[1] if len(args) > 1 else "40"
temp   = args[2] if len(args) > 2 else "0.0"

here = os.path.dirname(os.path.abspath(__file__))
models = os.path.join(here, "..", "models", "granite")
tok = Tokenizer.from_file(os.path.join(models, "tokenizer.json"))

text = prompt
if chat:
    text = (f"<|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>\n"
            f"<|start_of_role|>assistant<|end_of_role|>")

ids = tok.encode(text).ids
r = subprocess.run([os.path.join(here, "granite_gen"),
                    os.path.join(models, "granite.weights"), n_new, temp],
                   input=" ".join(map(str, ids)), capture_output=True, text=True)
sys.stderr.write(r.stderr)
if r.returncode != 0:
    sys.exit(r.returncode)

gen = [int(x) for x in r.stdout.split()]
print("PROMPT :", prompt)
print("OUTPUT :", tok.decode(gen) if chat else tok.decode(ids + gen))
