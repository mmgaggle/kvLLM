/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Parity test: HIP forward vs CPU reference on identical fp32 weights.
 * Passes when the max abs logit difference is within tolerance.
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "kmodel.h"

/* IEEE half (raw bits) -> float, portable host-side decode. */
static float
h2f(uint16_t h)
{
	uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff, f;
	if (exp == 0) {
		if (man == 0) {
			f = sign << 31;
		} else {
			exp = 127 - 15 + 1;
			while (!(man & 0x400)) { man <<= 1; exp--; }
			man &= 0x3ff;
			f = (sign << 31) | (exp << 23) | (man << 13);
		}
	} else if (exp == 0x1f) {
		f = (sign << 31) | (0xffu << 23) | (man << 13);
	} else {
		f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
	}
	float out;
	memcpy(&out, &f, 4);
	return out;
}

int
main(void)
{
	KCfg cfg = ktiny_cfg();
	HostModel m;
	host_model_init_random(m, cfg, 0xC0FFEEULL);

	const uint32_t tokens[] = { 7, 42, 3, 19, 60, 1, 33, 5, 12, 50 };
	int ntok = (int)(sizeof(tokens) / sizeof(tokens[0]));

	std::vector<float> ref, ref_kv;
	cpu_forward_logits(m, tokens, ntok, ref, &ref_kv);

	HipModel *hm = hip_upload(m);
	std::vector<float> gpu(cfg.vocab);
	std::vector<uint16_t> gpu_kv(ref_kv.size());
	hip_forward_logits(hm, tokens, ntok, gpu.data(), gpu_kv.data());
	hip_free(hm);

	float maxabs = 0.0f, maxrel = 0.0f;
	int argmax_ref = 0, argmax_gpu = 0;
	for (int i = 0; i < cfg.vocab; i++) {
		float d = fabsf(ref[i] - gpu[i]);
		if (d > maxabs)
			maxabs = d;
		float den = fabsf(ref[i]) + 1e-6f;
		if (d / den > maxrel)
			maxrel = d / den;
		if (ref[i] > ref[argmax_ref]) argmax_ref = i;
		if (gpu[i] > gpu[argmax_gpu]) argmax_gpu = i;
	}

	printf("forward parity (HIP vs CPU ref)\n");
	printf("  cfg: hidden=%d layers=%d heads=%d kv_heads=%d head_dim=%d vocab=%d\n",
	       cfg.hidden, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.vocab);
	printf("  ntok=%d  max|abs|=%.3e  max rel=%.3e\n", ntok, maxabs, maxrel);
	printf("  argmax: ref=%d gpu=%d\n", argmax_ref, argmax_gpu);
	printf("  ref[0..4]: % .5f % .5f % .5f % .5f\n", ref[0], ref[1], ref[2], ref[3]);
	printf("  gpu[0..4]: % .5f % .5f % .5f % .5f\n", gpu[0], gpu[1], gpu[2], gpu[3]);

	/* fp16 KV blocks (GPU) vs fp32 reference KV, in block layout. */
	float kv_maxabs = 0.0f;
	for (size_t i = 0; i < ref_kv.size(); i++) {
		float d = fabsf(ref_kv[i] - h2f(gpu_kv[i]));
		if (d > kv_maxabs)
			kv_maxabs = d;
	}
	int npages = kv_num_pages(ntok, cfg.page_tokens);
	printf("  KV blocks: pages=%d  block_elems=%zu  fp16 max|abs|=%.3e\n",
	       npages, kv_block_elems(cfg), kv_maxabs);

	const float logit_tol = 1e-3f;   /* fp32 forward */
	const float kv_tol    = 5e-3f;   /* fp16 storage round-trip */
	bool ok = (maxabs < logit_tol) && (argmax_ref == argmax_gpu) &&
		  (kv_maxabs < kv_tol) && (ref_kv.size() > 0);
	printf("\n%s (logit_tol=%.0e kv_tol=%.0e)\n", ok ? "PASS" : "FAIL",
	       logit_tol, kv_tol);
	return ok ? 0 : 1;
}
