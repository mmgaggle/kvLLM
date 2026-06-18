/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Warm-start: a cache HIT loads the content-addressed fp16 prefix KV straight
 * into the cache and SKIPS its prefill. The generated tokens must match the cold
 * path (full prefill). This is the content-addressed compute-skip — the thesis:
 * a shared prefix is never recomputed.
 */
#include <cstdint>
#include <cstdio>
#include <vector>
#include "gpt2.h"

int
main(void)
{
	Gpt2 *g = gpt2_load("models/gpt2/gpt2.weights");
	if (!g) { fprintf(stderr, "load failed\n"); return 1; }

	std::vector<uint32_t> prompt = { 464, 3625, 286, 1204, 318, 257, 1256, 286, 1257, 13 };
	int ntok = (int)prompt.size(), n_new = 24, P = 16;

	/* cold: full prefill + decode */
	Gpt2KV *kv1 = gpt2_kv_create(g, 2048);
	std::vector<uint32_t> p1 = prompt, cold;
	gpt2_generate_cached(g, kv1, p1, n_new, cold);
	gpt2_kv_free(kv1);

	/* the content-addressed KV blocks the store would hold for this prompt */
	int npages = (ntok + P - 1) / P;
	std::vector<uint16_t> blocks((size_t)npages * gpt2_kv_block_elems(g, P));
	std::vector<float> dummy(gpt2_vocab(g));
	gpt2_forward_logits(g, prompt.data(), ntok, dummy.data(), blocks.data(), P);

	/* warm: load ntok-1 cached positions, recompute only the last prompt token */
	int n_load = ntok - 1;
	Gpt2KV *kv2 = gpt2_kv_create(g, 2048);
	std::vector<uint32_t> p2 = prompt, warm;
	gpt2_generate_warm(g, kv2, blocks.data(), n_load, P, p2, n_new, warm);
	gpt2_kv_free(kv2);

	int match = 0;
	while (match < (int)cold.size() && match < (int)warm.size() && cold[match] == warm[match])
		match++;
	bool same = (cold == warm);

	printf("warm-start (content-addressed prefix skip)\n");
	printf("  prompt=%d  prefill tokens SKIPPED via cache=%d  gen=%d\n",
	       ntok, n_load, n_new);
	printf("  cold[0..6]:");
	for (int i = 0; i < 6; i++) printf(" %u", cold[i]);
	printf("\n  warm[0..6]:");
	for (int i = 0; i < 6; i++) printf(" %u", warm[i]);
	printf("\n  matching generated tokens: %d/%d\n", match, (int)cold.size());

	gpt2_free(g);
	printf("\n%s\n", same ? "PASS" : "FAIL");
	return same ? 0 : 1;
}
