/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Integration: greedy decode end to end, with KV flowing into the
 * content-addressed store.
 *
 *   prompt bytes -> tokenize -> [ GPU forward -> logits + fp16 KV blocks
 *                                -> content-address each page (prefix Merkle
 *                                   hash) -> store/hit in the kvdev-shaped KV
 *                                   store -> argmax -> append ] x N
 *
 * This is the first time the GPU compute (gpu/kmodel) and the storage spine
 * (src/chash + src/kvstore, the real engine components) run together. Weights
 * are synthetic, so the generated tokens are gibberish — but the whole pipeline
 * is real, and KV blocks are shared across runs by content.
 *
 * This recomputes the full sequence each step (O(n^2)); the continuation forward uses
 * the cached prefix KV to compute only the suffix.
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "kmodel.h"
#include "chash.h"     /* extern "C" */
#include "kvstore.h"   /* extern "C" */

struct GenStats {
	int generated;
	int page_hits;
	int page_misses;
};

static int
argmax(const float *v, int n)
{
	int b = 0;
	for (int i = 1; i < n; i++)
		if (v[i] > v[b])
			b = i;
	return b;
}

/* Greedy-decode n_new tokens, content-addressing every page's KV against ks. */
static GenStats
generate(HipModel *hm, const KCfg &cfg, kllm_kvstore *ks, kllm_key seed,
	 std::vector<uint32_t> &toks, int n_new, std::vector<uint32_t> *out)
{
	GenStats st = { 0, 0, 0 };
	int P = cfg.page_tokens;
	size_t be = kv_block_elems(cfg);           /* fp16 elems per page block */
	std::vector<float> logits(cfg.vocab);

	for (int s = 0; s < n_new; s++) {
		int ntok = (int)toks.size();
		int npages = kv_num_pages(ntok, P);
		std::vector<uint16_t> kv((size_t)npages * be);

		hip_forward_logits(hm, toks.data(), ntok, logits.data(), kv.data());

		kllm_key parent = seed;
		for (int p = 0; p < npages; p++) {
			int p0 = p * P;
			int pn = (ntok - p0) < P ? (ntok - p0) : P;
			kllm_key key = kllm_chash_chain(parent, &toks[p0],
							(size_t)pn * sizeof(uint32_t));
			if (kllm_kvstore_exist(ks, key)) {
				st.page_hits++;
			} else {
				kllm_kvstore_store(ks, key, &kv[(size_t)p * be],
						   (uint32_t)(be * sizeof(uint16_t)));
				st.page_misses++;
			}
			parent = key;
		}

		uint32_t next = (uint32_t)argmax(logits.data(), cfg.vocab);
		toks.push_back(next);
		if (out)
			out->push_back(next);
		st.generated++;
	}
	return st;
}

static void
tokenize(const char *s, std::vector<uint32_t> &toks)
{
	toks.clear();
	for (const unsigned char *p = (const unsigned char *)s; *p; p++)
		toks.push_back(*p);  /* byte-level, matches the engine */
}

static void
run(HipModel *hm, const KCfg &cfg, kllm_kvstore *ks, kllm_key seed,
    const char *label, const char *prompt, int n_new)
{
	std::vector<uint32_t> toks, gen;
	tokenize(prompt, toks);
	GenStats st = generate(hm, cfg, ks, seed, toks, n_new, &gen);

	printf("%-22s prompt=\"%s\"  gen=%d  page hits=%d misses=%d  store=%zu blocks\n",
	       label, prompt, st.generated, st.page_hits, st.page_misses,
	       kllm_kvstore_count(ks));
	printf("    tokens:");
	for (uint32_t t : gen)
		printf(" %u", t);
	printf("\n");
}

int
main(void)
{
	KCfg cfg = ktiny_cfg();
	HostModel m;
	host_model_init_random(m, cfg, 0xC0FFEEULL);
	HipModel *hm = hip_upload(m);

	/* namespace seed = compute-context identity (model|dtype|...) */
	const char *ctx = "ktiny|v0|fp16";
	kllm_key seed = kllm_chash(ctx, strlen(ctx));
	kllm_kvstore *ks = kllm_kvstore_create();

	char nshex[2 * KLLM_KEY_LEN + 1];
	kllm_key_hex(seed, nshex);
	printf("kLLM — greedy decode + content-addressed KV\n");
	printf("namespace seed: %s  page_tokens=%d  block=%zu B\n\n",
	       nshex, cfg.page_tokens, kv_block_elems(cfg) * sizeof(uint16_t));

	run(hm, cfg, ks, seed, "1) cold",            "hello world", 8);
	run(hm, cfg, ks, seed, "2) same prompt",     "hello world", 8);
	run(hm, cfg, ks, seed, "3) shared prefix",   "hello there", 8);

	kllm_kvstore_destroy(ks);
	hip_free(hm);
	printf("\nOK\n");
	return 0;
}
