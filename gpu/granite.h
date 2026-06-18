/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * IBM Granite 4.0 (dense transformer nano) in HIP. A real instruct-tuned model:
 * RMSNorm, GQA, rotate-half RoPE, SwiGLU (fused gate/up), tied embeddings, plus
 * Granite's scalar multipliers (embedding/attention/residual/logits). Mirrors
 * the gpt2.* API so the serving daemon and KV-cache logic are shared in shape.
 *
 * Weights come from gpu/export_granite.py (flat fp32). Tokenization is userspace.
 */
#ifndef KLLM_GRANITE_H
#define KLLM_GRANITE_H

#include <cstdint>
#include <vector>

struct Granite;  /* opaque: weights in VRAM */

Granite *granite_load(const char *weights_path);
void granite_free(Granite *g);
int  granite_vocab(const Granite *g);

/* fp16 elements in one KV page block: 2 * n_layers * page_tokens * (kv_heads*head_dim). */
size_t granite_kv_block_elems(const Granite *g, int page_tokens);

/* Forward over toks[0:ntok); last-position logits. If kv_blocks_host != null,
 * also writes the fp16 KV page blocks for the whole sequence. */
void granite_forward_logits(Granite *g, const uint32_t *toks, int ntok, float *logits_host,
			    uint16_t *kv_blocks_host = nullptr, int page_tokens = 16);

void granite_generate(Granite *g, std::vector<uint32_t> &toks, int n_new,
		      std::vector<uint32_t> &out);

/* Incremental KV cache. */
struct GraniteKV;
GraniteKV *granite_kv_create(Granite *g, int maxctx);
void granite_kv_free(GraniteKV *kv);

void granite_generate_cached(Granite *g, GraniteKV *kv, std::vector<uint32_t> &toks,
			     int n_new, std::vector<uint32_t> &out,
			     float temp = 0.0f, uint64_t seed = 0);

/* Warm-start: load n_load cached prefix positions, skip their prefill. */
void granite_generate_warm(Granite *g, GraniteKV *kv, const uint16_t *prefix_blocks,
			   int n_load, int page_tokens, std::vector<uint32_t> &toks,
			   int n_new, std::vector<uint32_t> &out,
			   float temp = 0.0f, uint64_t seed = 0);

/* Pack the KV cache [0,len) into fp16 page blocks for content-addressed storage. */
void granite_kv_to_blocks(Granite *g, GraniteKV *kv, int page_tokens, uint16_t *blocks_host);

#endif /* KLLM_GRANITE_H */
