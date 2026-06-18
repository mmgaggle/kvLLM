/* SPDX-License-Identifier: BSD-3-Clause */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "engine.h"
#include "tokenizer.h"
#include "chash.h"
#include "kvstore.h"

const struct kllm_model_cfg KLLM_MODEL_TINY = {
	.name        = "tiny",
	.weights_ver = "v0",
	.dtype       = "fp16",
	.page_tokens = 8,
	.num_layers  = 4,
	.num_kv_heads = 4,
	.head_dim    = 64,
	.dtype_bytes = 2,
};

struct kllm_engine {
	struct kllm_model_cfg     cfg;
	struct kllm_kvstore      *ks;
	struct kllm_key           ns_seed;   /* compute-context identity */
	uint64_t                  block_bytes;
	uint8_t                  *scratch;    /* one block, reused per page */
	struct kllm_kv_backend    be;         /* producer seam (stub / HIP) */
};

/*
 * Stub backend: the GPU forward pass stand-in. Produces a deterministic
 * payload from the block key so recomputing the same prefix yields identical
 * bytes (content addressing stays consistent). Ignores tokens/pos.
 */
static void
stub_produce(void *ctx, struct kllm_key key, const uint32_t *tokens,
	     size_t ntok, uint64_t pos, uint8_t *out, uint64_t block_bytes)
{
	(void)ctx; (void)tokens; (void)ntok; (void)pos;
	uint8_t seed = 0;
	for (int i = 0; i < KLLM_KEY_LEN; i++)
		seed ^= key.b[i];
	memset(out, seed, block_bytes);
}

const struct kllm_kv_backend KLLM_KV_BACKEND_STUB = {
	.name = "stub",
	.produce = stub_produce,
	.ctx = NULL,
};

uint64_t
kllm_engine_block_bytes(const struct kllm_engine *eng)
{
	return eng->block_bytes;
}

void
kllm_engine_namespace_hex(const struct kllm_engine *eng, char *out)
{
	kllm_key_hex(eng->ns_seed, out);
}

size_t
kllm_engine_block_count(const struct kllm_engine *eng)
{
	return kllm_kvstore_count(eng->ks);
}

struct kllm_engine *
kllm_engine_create(const struct kllm_model_cfg *cfg)
{
	return kllm_engine_create_be(cfg, &KLLM_KV_BACKEND_STUB);
}

struct kllm_engine *
kllm_engine_create_be(const struct kllm_model_cfg *cfg,
		      const struct kllm_kv_backend *be)
{
	struct kllm_engine *eng = calloc(1, sizeof(*eng));
	if (!eng)
		return NULL;

	eng->cfg = *cfg;
	eng->be = *be;
	eng->ks = kllm_kvstore_create();
	if (!eng->ks) {
		free(eng);
		return NULL;
	}

	/* namespace seed = H(model ‖ weights_ver ‖ dtype). Different runtimes
	 * produce different seeds, so their blocks never share a key. */
	char ctx[256];
	int n = snprintf(ctx, sizeof(ctx), "%s|%s|%s|pt=%u|L=%u|kvh=%u|hd=%u|db=%u",
			 cfg->name, cfg->weights_ver, cfg->dtype, cfg->page_tokens,
			 cfg->num_layers, cfg->num_kv_heads, cfg->head_dim,
			 cfg->dtype_bytes);
	eng->ns_seed = kllm_chash(ctx, (size_t)n);

	eng->block_bytes = 2ull * cfg->num_layers * cfg->page_tokens *
			   cfg->num_kv_heads * cfg->head_dim * cfg->dtype_bytes;

	eng->scratch = malloc(eng->block_bytes);
	if (!eng->scratch) {
		kllm_kvstore_destroy(eng->ks);
		free(eng);
		return NULL;
	}
	return eng;
}

void
kllm_engine_destroy(struct kllm_engine *eng)
{
	if (!eng)
		return;
	free(eng->scratch);
	kllm_kvstore_destroy(eng->ks);
	free(eng);
}

int
kllm_engine_run(struct kllm_engine *eng, const void *prompt, size_t len,
		struct kllm_run_stats *out)
{
	memset(out, 0, sizeof(*out));

	uint32_t *tokens = malloc((len ? len : 1) * sizeof(uint32_t));
	if (!tokens)
		return -ENOMEM;

	size_t ntok = kllm_tokenize(prompt, len, tokens, len);
	out->tokens = (uint32_t)ntok;

	uint32_t pt = eng->cfg.page_tokens;
	struct kllm_key parent = eng->ns_seed;
	int rc = 0;

	for (size_t off = 0; off < ntok; off += pt) {
		size_t page_n = (ntok - off) < pt ? (ntok - off) : pt;
		struct kllm_key key =
			kllm_chash_chain(parent, &tokens[off],
					 page_n * sizeof(uint32_t));
		out->pages++;

		if (kllm_kvstore_exist(eng->ks, key)) {
			uint32_t vlen = 0;
			rc = kllm_kvstore_retrieve(eng->ks, key, eng->scratch,
						   (uint32_t)eng->block_bytes, &vlen);
			if (rc)
				break;
			out->hits++;
			out->bytes_fetched += vlen;
		} else {
			eng->be.produce(eng->be.ctx, key, &tokens[off], page_n,
					(uint64_t)off, eng->scratch,
					eng->block_bytes);
			rc = kllm_kvstore_store(eng->ks, key, eng->scratch,
						(uint32_t)eng->block_bytes);
			if (rc)
				break;
			out->misses++;
			out->bytes_computed += eng->block_bytes;
		}
		parent = key;
	}

	free(tokens);
	return rc;
}
