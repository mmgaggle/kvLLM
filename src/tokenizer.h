/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Tokenizer: byte-level. Each input byte maps to one token id in
 * [0,255]. This is a real, deterministic tokenization (cf. ByT5 / GPT-2
 * byte fallback) — sufficient to exercise content-addressed prefix sharing —
 * the model-correct BPE lives with the GPU models, in userspace.
 *
 * Lives in userspace by design: the eBPF verifier fights data-dependent BPE
 * merge loops, and tokenization isn't the bottleneck.
 */
#ifndef KLLM_TOKENIZER_H
#define KLLM_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

/*
 * Tokenize up to max_tokens tokens from text[0..len). Writes token ids into
 * tokens[] and returns the number produced (== min(len, max_tokens)).
 */
size_t kllm_tokenize(const void *text, size_t len, uint32_t *tokens, size_t max_tokens);

#endif /* KLLM_TOKENIZER_H */
