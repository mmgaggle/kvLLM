/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Cache-invariant test — the most important correctness check in kLLM.
 *
 * Full recompute over tokens[0:ntok] (fp32 KV) must produce the same last-token
 * logits as the continuation forward, which uses the cached fp16 prefix KV for
 * [0:prefix_len) and computes only the suffix [prefix_len:ntok). If this holds,
 * loading shared KV is safe; if the argmax ever differs, content-addressed KV
 * sharing would silently corrupt generation.
 *
 * The two paths are NOT bit-identical by construction: the continuation reads
 * the prefix KV as fp16, the full recompute keeps it fp32. So the bar is: agree
 * within fp16 tolerance, and — decisively for greedy decode — same argmax.
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "kmodel.h"

static int
argmax(const std::vector<float> &v)
{
	int b = 0;
	for (int i = 1; i < (int)v.size(); i++)
		if (v[i] > v[b])
			b = i;
	return b;
}

static bool
check_prefix(HipModel *hm, const KCfg &cfg, const uint32_t *tokens, int ntok,
	     const std::vector<uint16_t> &kv, int prefix_len)
{
	int npp = prefix_len / cfg.page_tokens;
	size_t page_elems = kv_block_elems(cfg);

	std::vector<float> full(cfg.vocab), cont(cfg.vocab);
	/* recompute the full forward's last-token logits */
	hip_forward_logits(hm, tokens, ntok, full.data());
	/* continuation using the first npp cached pages as prefix context */
	hip_forward_cont(hm, tokens, ntok, prefix_len,
			 kv.data() /* pages [0, npp) */, cont.data());
	(void)page_elems;

	float maxabs = 0.0f;
	for (int i = 0; i < cfg.vocab; i++)
		maxabs = fmaxf(maxabs, fabsf(full[i] - cont[i]));
	int af = argmax(full), ac = argmax(cont);
	bool ok = (af == ac) && (maxabs < 5e-2f);
	printf("  prefix_len=%-3d (pages=%d, suffix=%d): max|abs|=%.3e  argmax full=%d cont=%d  %s\n",
	       prefix_len, npp, ntok - prefix_len, maxabs, af, ac, ok ? "ok" : "MISMATCH");
	return ok;
}

int
main(void)
{
	KCfg cfg = ktiny_cfg();
	HostModel m;
	host_model_init_random(m, cfg, 0xC0FFEEULL);
	HipModel *hm = hip_upload(m);

	const uint32_t tokens[] = { 7, 42, 3, 19, 60, 1, 33, 5, 12, 50, 8, 21 };
	int ntok = (int)(sizeof(tokens) / sizeof(tokens[0]));

	/* produce the fp16 KV blocks for the whole sequence once */
	std::vector<float> logits(cfg.vocab);
	std::vector<uint16_t> kv((size_t)kv_num_pages(ntok, cfg.page_tokens) * kv_block_elems(cfg));
	hip_forward_logits(hm, tokens, ntok, logits.data(), kv.data());

	printf("cache invariant (cont w/ fp16 cached prefix == full recompute)\n");
	printf("  ntok=%d page_tokens=%d\n", ntok, cfg.page_tokens);

	bool ok = true;
	for (int pl = 0; pl <= ntok - 1; pl += cfg.page_tokens)
		ok &= check_prefix(hm, cfg, tokens, ntok, kv, pl);

	hip_free(hm);
	printf("\n%s\n", ok ? "PASS" : "FAIL");
	return ok ? 0 : 1;
}
