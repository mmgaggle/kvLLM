/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * The incremental KV cache must produce IDENTICAL greedy tokens to the
 * O(n^2) recompute path, and do it faster (O(n) decode). Correctness first,
 * speed second.
 */
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <vector>
#include "gpt2.h"

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b)
{
	return std::chrono::duration<double, std::milli>(b - a).count();
}

int
main(void)
{
	Gpt2 *g = gpt2_load("models/gpt2/gpt2.weights");
	if (!g) {
		fprintf(stderr, "load failed\n");
		return 1;
	}

	/* arbitrary valid prompt token ids (correctness is meaning-independent) */
	std::vector<uint32_t> prompt = { 464, 3625, 286, 1204, 318, 257, 1256, 286, 1257, 13 };
	int n_new = 128;

	/* warm up the GPU (JIT, allocators) so timing is fair */
	{ std::vector<uint32_t> w = prompt, o; gpt2_generate(g, w, 1, o); }

	std::vector<uint32_t> p1 = prompt, ref;
	auto t0 = clk::now();
	gpt2_generate(g, p1, n_new, ref);             /* O(n^2) recompute */
	auto t1 = clk::now();

	std::vector<uint32_t> p2 = prompt, fast;
	Gpt2KV *kv = gpt2_kv_create(g, 2048);
	auto t2 = clk::now();
	gpt2_generate_cached(g, kv, p2, n_new, fast); /* O(n) incremental cache */
	auto t3 = clk::now();
	gpt2_kv_free(kv);

	bool same = (ref == fast);

	/* token-forward units of compute: recompute forwards the whole growing
	 * sequence each step; the cache prefills once then forwards 1 token/step. */
	long P = (long)prompt.size();
	long units_recompute = 0;
	for (int i = 0; i < n_new; i++)
		units_recompute += P + i;
	long units_cache = P + n_new;

	printf("incremental KV cache (prompt=%ld, gen=%d)\n", P, n_new);
	printf("  compute (token-forwards): recompute %ld  vs  cache %ld  -> %.1fx less\n",
	       units_recompute, units_cache, (double)units_recompute / units_cache);
	printf("  wall clock:  recompute %8.1f ms   cache %8.1f ms   (%.2fx)\n",
	       ms(t0, t1), ms(t2, t3), ms(t0, t1) / ms(t2, t3));
	printf("  (gpt2-small batch-1 is kernel-launch-bound; cache's win is the\n");
	printf("   compute reduction above, which dominates at scale / with fusion)\n");
	printf("  identical tokens: %s\n", same ? "yes" : "NO");
	printf("  ref[0..6] :");
	for (int i = 0; i < 6 && i < (int)ref.size(); i++) printf(" %u", ref[i]);
	printf("\n  fast[0..6]:");
	for (int i = 0; i < 6 && i < (int)fast.size(); i++) printf(" %u", fast[i]);
	printf("\n\n%s\n", same ? "PASS" : "FAIL");

	gpt2_free(g);
	return same ? 0 : 1;
}
