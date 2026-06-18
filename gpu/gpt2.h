/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * GPT-2 (124M) in HIP. A real, trained model so the
 * pipeline produces real English instead of gibberish. Different architecture
 * from the Llama-style kmodel (LayerNorm, learned positions, GELU, combined
 * QKV, tied lm_head) — but no RoPE, so no convention-matching risk.
 *
 * Weights come from gpu/export_gpt2.py (flat fp32). Tokenization is done in
 * userspace by gpu/run_gpt2.py (the real GPT-2 BPE), matching the locked
 * "tokenizer in userspace" decision.
 */
#ifndef KLLM_GPT2_H
#define KLLM_GPT2_H

#include <cstdint>
#include <vector>

struct Gpt2;  /* opaque: weights in VRAM */

Gpt2 *gpt2_load(const char *weights_path);
void gpt2_free(Gpt2 *g);
int  gpt2_vocab(const Gpt2 *g);
int  gpt2_layers(const Gpt2 *g);
int  gpt2_embed(const Gpt2 *g);

/* fp16 elements in one KV page block: 2 (K,V) * n_layers * page_tokens * n_embd. */
size_t gpt2_kv_block_elems(const Gpt2 *g, int page_tokens);

/* Forward over toks[0:ntok); writes vocab logits for the last position. If
 * kv_blocks_host != null, also writes the fp16 KV page blocks (raw uint16) for
 * the whole sequence: ceil(ntok/page_tokens) * gpt2_kv_block_elems entries. */
void gpt2_forward_logits(Gpt2 *g, const uint32_t *toks, int ntok, float *logits_host,
			 uint16_t *kv_blocks_host = nullptr, int page_tokens = 16);

/* Greedy-generate n_new tokens, appending each to toks and to out. */
void gpt2_generate(Gpt2 *g, std::vector<uint32_t> &toks, int n_new,
		   std::vector<uint32_t> &out);

/* Incremental KV cache: O(n) decode instead of O(n^2) recompute. */
struct Gpt2KV;
Gpt2KV *gpt2_kv_create(Gpt2 *g, int maxctx);
void gpt2_kv_free(Gpt2KV *kv);

/* Generate using the incremental KV cache: prefill the prompt, then decode one
 * token at a time. temp<=0 is greedy (deterministic); temp>0 samples from
 * softmax(logits/temp) with the given seed. */
void gpt2_generate_cached(Gpt2 *g, Gpt2KV *kv, std::vector<uint32_t> &toks,
			  int n_new, std::vector<uint32_t> &out,
			  float temp = 0.0f, uint64_t seed = 0);

/* Warm-start: load n_load cached prefix positions (fp16 page blocks, as written
 * by gpt2_forward_logits) straight into the KV cache — SKIPPING their prefill —
 * then process the remaining prompt tokens and generate. This is the
 * content-addressed compute-skip: a shared prefix is never recomputed. */
void gpt2_generate_warm(Gpt2 *g, Gpt2KV *kv, const uint16_t *prefix_blocks,
			int n_load, int page_tokens, std::vector<uint32_t> &toks,
			int n_new, std::vector<uint32_t> &out,
			float temp = 0.0f, uint64_t seed = 0);

/* Pack the KV cache [0,len) into fp16 page blocks (host buffer sized
 * ceil(len/page_tokens) * gpt2_kv_block_elems) for content-addressed storage. */
void gpt2_kv_to_blocks(Gpt2 *g, Gpt2KV *kv, int page_tokens, uint16_t *blocks_host);

#endif /* KLLM_GPT2_H */

