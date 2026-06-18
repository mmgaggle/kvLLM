/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * kLLM engine: the request pipeline that ties the head together.
 *
 *   prompt bytes -> tokenize -> split into fixed-size pages -> per page,
 *   content-address via prefix Merkle hash -> hit-or-produce against the
 *   shared KV store.
 *
 * The KV *producer* is pluggable: a default stub (deterministic bytes per block)
 * stands in for a GPU paged-attention forward pass. The engine proves the
 * addressing/sharing thread end to end and measures a cross-request
 * prefix-cache hit.
 *
 * Compute context (model, weights version, dtype, ...) is folded into a
 * per-namespace seed key, realizing the "namespace encodes context, key is a
 * pure prefix hash" decision: blocks from different runtimes can never collide.
 */
#ifndef KLLM_ENGINE_H
#define KLLM_ENGINE_H

#include <stdint.h>
#include <stddef.h>
#include "chash.h"   /* struct kllm_key */

#ifdef __cplusplus
extern "C" {
#endif

struct kllm_model_cfg {
	const char *name;         /* identity: part of the namespace seed */
	const char *weights_ver;
	const char *dtype;
	uint32_t    page_tokens;  /* tokens per KV page (paged attention block) */
	/* dims used to size a KV block: 2 (K+V) * layers * page_tokens *
	 * kv_heads * head_dim * dtype_bytes */
	uint32_t    num_layers;
	uint32_t    num_kv_heads;
	uint32_t    head_dim;
	uint32_t    dtype_bytes;
};

/* A small reference config for the prototype. */
extern const struct kllm_model_cfg KLLM_MODEL_TINY;

/*
 * KV-block producer backend — the seam where the GPU forward pass lands.
 *
 * The engine calls produce() on a cache miss to materialize one page's KV
 * bytes. The default KLLM_KV_BACKEND_STUB produces deterministic bytes (CPU, no
 * model); a HIP backend would compute real K/V.
 *
 * NOTE: this signature produces a single page in isolation, which is correct
 * for the stub. A real forward pass needs the cached-prefix KV as attention
 * context and emits logits for sampling, and would extend this struct
 * (append only) accordingly.
 */
struct kllm_kv_backend {
	const char *name;
	/*
	 * Fill out[0..block_bytes) with the KV for the page identified by `key`,
	 * covering tokens[0..ntok) at sequence position `pos`. ctx is backend
	 * private state.
	 */
	void (*produce)(void *ctx, struct kllm_key key,
			const uint32_t *tokens, size_t ntok, uint64_t pos,
			uint8_t *out, uint64_t block_bytes);
	void *ctx;
};

/* The deterministic CPU stub. */
extern const struct kllm_kv_backend KLLM_KV_BACKEND_STUB;

struct kllm_engine;

/* Create with the default (stub) producer backend. */
struct kllm_engine *kllm_engine_create(const struct kllm_model_cfg *cfg);

/* Create with an explicit producer backend (e.g. a HIP backend). */
struct kllm_engine *kllm_engine_create_be(const struct kllm_model_cfg *cfg,
					  const struct kllm_kv_backend *be);
void kllm_engine_destroy(struct kllm_engine *eng);

struct kllm_run_stats {
	uint32_t tokens;
	uint32_t pages;
	uint32_t hits;          /* pages served from cache */
	uint32_t misses;        /* pages computed (stub) + stored */
	uint64_t bytes_fetched; /* KV bytes read from the store on hits */
	uint64_t bytes_computed;/* KV bytes produced+stored on misses */
};

/*
 * Run one request against the shared KV store. Returns 0 or -errno; fills *out.
 * The store persists across calls, so a later request sharing a token prefix
 * hits on the shared blocks.
 */
int kllm_engine_run(struct kllm_engine *eng, const void *prompt, size_t len,
		    struct kllm_run_stats *out);

/* Bytes in one KV page block for this model. */
uint64_t kllm_engine_block_bytes(const struct kllm_engine *eng);

/* Hex of the namespace seed key (compute-context identity). out >= 33 bytes. */
void kllm_engine_namespace_hex(const struct kllm_engine *eng, char *out);

/* Total distinct blocks currently held by the engine's store. */
size_t kllm_engine_block_count(const struct kllm_engine *eng);

#ifdef __cplusplus
}
#endif

#endif /* KLLM_ENGINE_H */
