/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Spine test. Proves the content-addressing thread end to end without
 * needing /dev/cuse: tokenize -> page -> Merkle-hash -> hit-or-produce against
 * the shared in-memory KV store. The thesis under test is cross-request prefix
 * sharing.
 */
#include <stdio.h>
#include <string.h>
#include "engine.h"

static int failures;

static void
check(int cond, const char *what)
{
	printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
	if (!cond)
		failures++;
}

static struct kllm_run_stats
run(struct kllm_engine *eng, const char *prompt)
{
	struct kllm_run_stats s;
	int rc = kllm_engine_run(eng, prompt, strlen(prompt), &s);
	printf("  run \"%s\": tokens=%u pages=%u hits=%u misses=%u "
	       "fetched=%llu computed=%llu (rc=%d)\n",
	       prompt, s.tokens, s.pages, s.hits, s.misses,
	       (unsigned long long)s.bytes_fetched,
	       (unsigned long long)s.bytes_computed, rc);
	return s;
}

int
main(void)
{
	struct kllm_engine *eng = kllm_engine_create(&KLLM_MODEL_TINY);
	if (!eng) {
		fprintf(stderr, "engine create failed\n");
		return 2;
	}

	char ns[33];
	kllm_engine_namespace_hex(eng, ns);
	printf("kLLM spine test\n");
	printf("  namespace seed: %s  block_bytes=%llu page_tokens=%u\n\n",
	       ns, (unsigned long long)kllm_engine_block_bytes(eng),
	       KLLM_MODEL_TINY.page_tokens);

	/* 1. Cold run: every page is a miss. */
	printf("1) cold run\n");
	struct kllm_run_stats a = run(eng, "The quick brown fox jumps");
	check(a.misses == a.pages && a.hits == 0, "cold run is all misses");
	check(a.pages > 1, "prompt spans multiple pages");

	/* 2. Identical run: every page hits the shared blocks. */
	printf("2) identical re-run\n");
	struct kllm_run_stats b = run(eng, "The quick brown fox jumps");
	check(b.hits == b.pages && b.misses == 0, "identical re-run is all hits");
	check(kllm_engine_block_count(eng) == a.misses,
	      "no new blocks stored on identical re-run");

	/* 3. Shared prefix, divergent suffix: prefix pages hit, the rest miss. */
	printf("3) shared-prefix divergent run\n");
	struct kllm_run_stats c = run(eng, "The quick brown fox sleeps");
	check(c.hits > 0, "shared prefix produces hits");
	check(c.misses > 0, "divergent suffix produces misses");
	check(c.hits + c.misses == c.pages, "hits + misses account for all pages");

	/* 4. Namespace isolation: a different compute context shares nothing. */
	printf("4) namespace isolation\n");
	struct kllm_model_cfg other = KLLM_MODEL_TINY;
	other.dtype = "bf16";  /* different runtime => different namespace seed */
	struct kllm_engine *eng2 = kllm_engine_create(&other);
	char ns2[33];
	kllm_engine_namespace_hex(eng2, ns2);
	check(strcmp(ns, ns2) != 0, "different dtype yields a different namespace seed");
	struct kllm_run_stats d = run(eng2, "The quick brown fox jumps");
	check(d.misses == d.pages && d.hits == 0,
	      "same prompt in a fresh namespace shares nothing");
	kllm_engine_destroy(eng2);

	kllm_engine_destroy(eng);
	printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "OK",
	       failures, failures == 1 ? "" : "s");
	return failures ? 1 : 0;
}
