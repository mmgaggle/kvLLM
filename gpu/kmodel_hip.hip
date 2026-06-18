// SPDX-License-Identifier: BSD-3-Clause
// HIP forward pass (fp32) — mirrors gpu/kmodel_ref.cpp op for op.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "kmodel.h"

#define HIP_CHECK(expr)                                                       \
	do {                                                                  \
		hipError_t _e = (expr);                                       \
		if (_e != hipSuccess) {                                       \
			fprintf(stderr, "HIP error %s at %s:%d\n",            \
				hipGetErrorString(_e), __FILE__, __LINE__);  \
			abort();                                             \
		}                                                            \
	} while (0)

#define BLK 256
#define NBLK(n) (((n) + BLK - 1) / BLK)
#define MAXSEQ 256  // attention scratch cap (test sequences are short)

// ---- device weights -------------------------------------------------------

struct DLayer {
	float *attn_norm, *wq, *wk, *wv, *wo, *ffn_norm, *wg, *wu, *wd;
};

struct HipModel {
	KCfg cfg;
	float *tok_emb, *final_norm, *lm_head;
	std::vector<DLayer> layers;
};

static float *
to_dev(const std::vector<float> &h)
{
	float *d = nullptr;
	HIP_CHECK(hipMalloc(&d, h.size() * sizeof(float)));
	HIP_CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(float), hipMemcpyHostToDevice));
	return d;
}

static void
dfree(void *p)
{
	if (p)
		(void)hipFree(p);  /* cleanup path: ignore status */
}

HipModel *
hip_upload(const HostModel &m)
{
	auto *hm = new HipModel();
	hm->cfg = m.cfg;
	hm->tok_emb = to_dev(m.tok_emb);
	hm->final_norm = to_dev(m.final_norm);
	hm->lm_head = to_dev(m.lm_head);
	hm->layers.resize(m.layers.size());
	for (size_t i = 0; i < m.layers.size(); i++) {
		const HostLayer &L = m.layers[i];
		DLayer &D = hm->layers[i];
		D.attn_norm = to_dev(L.attn_norm);
		D.wq = to_dev(L.wq); D.wk = to_dev(L.wk);
		D.wv = to_dev(L.wv); D.wo = to_dev(L.wo);
		D.ffn_norm = to_dev(L.ffn_norm);
		D.wg = to_dev(L.wg); D.wu = to_dev(L.wu); D.wd = to_dev(L.wd);
	}
	return hm;
}

void
hip_free(HipModel *hm)
{
	if (!hm)
		return;
	dfree(hm->tok_emb); dfree(hm->final_norm); dfree(hm->lm_head);
	for (auto &D : hm->layers) {
		dfree(D.attn_norm); dfree(D.wq); dfree(D.wk); dfree(D.wv);
		dfree(D.wo); dfree(D.ffn_norm); dfree(D.wg); dfree(D.wu); dfree(D.wd);
	}
	delete hm;
}

// ---- kernels (mirror kmodel_ref.cpp) --------------------------------------

__global__ void
k_embed(const float *tok_emb, const uint32_t *tok, float *x, int M, int H)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= M * H)
		return;
	int t = idx / H, i = idx % H;
	x[idx] = tok_emb[(size_t)tok[t] * H + i];
}

__global__ void
k_rmsnorm(const float *X, const float *w, float *Y, int M, int H, float eps)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= M)
		return;
	float ss = 0.0f;
	for (int i = 0; i < H; i++) {
		float v = X[t * H + i];
		ss += v * v;
	}
	float sc = 1.0f / sqrtf(ss / H + eps);
	for (int i = 0; i < H; i++)
		Y[t * H + i] = X[t * H + i] * sc * w[i];
}

// Y[M][N] = X[M][K] @ W[K][N]
__global__ void
k_mm(const float *X, const float *W, float *Y, int M, int K, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= M * N)
		return;
	int m = idx / N, n = idx % N;
	float a = 0.0f;
	for (int k = 0; k < K; k++)
		a += X[m * K + k] * W[k * N + n];
	Y[idx] = a;
}

// pos0 is the absolute position of row 0 (0 for a full forward, prefix_len for a
// suffix continuation).
__global__ void
k_rope(float *B, int M, int nh, int hd, float theta, int pos0)
{
	int half = hd / 2;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= M * nh * half)
		return;
	int p = idx % half;
	int hh = (idx / half) % nh;
	int t = idx / half / nh;
	float freq = powf(theta, -2.0f * p / hd);
	float ang = (pos0 + t) * freq;
	float c = cosf(ang), s = sinf(ang);
	float *base = B + (size_t)(t * nh + hh) * hd;
	float x0 = base[2 * p], x1 = base[2 * p + 1];
	base[2 * p]     = x0 * c - x1 * s;
	base[2 * p + 1] = x0 * s + x1 * c;
}

__global__ void
k_attn(const float *Q, const float *K, const float *V, float *O,
       int M, int nh, int nkv, int hd)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= M * nh)
		return;
	int h = idx % nh, t = idx / nh;
	int group = nh / nkv, kvh = h / group;
	float scale = 1.0f / sqrtf((float)hd);
	const float *qp = Q + (size_t)(t * nh + h) * hd;
	float sc[MAXSEQ];
	float maxv = -1e30f;
	for (int tp = 0; tp <= t; tp++) {
		const float *kp = K + (size_t)(tp * nkv + kvh) * hd;
		float d = 0.0f;
		for (int i = 0; i < hd; i++)
			d += qp[i] * kp[i];
		d *= scale;
		sc[tp] = d;
		if (d > maxv)
			maxv = d;
	}
	float sum = 0.0f;
	for (int tp = 0; tp <= t; tp++) {
		sc[tp] = expf(sc[tp] - maxv);
		sum += sc[tp];
	}
	float *op = O + (size_t)(t * nh + h) * hd;
	for (int i = 0; i < hd; i++)
		op[i] = 0.0f;
	for (int tp = 0; tp <= t; tp++) {
		const float *vp = V + (size_t)(tp * nkv + kvh) * hd;
		float w = sc[tp] / sum;
		for (int i = 0; i < hd; i++)
			op[i] += w * vp[i];
	}
}

__global__ void
k_add(float *X, const float *Y, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		X[i] += Y[i];
}

__global__ void
k_silumul(const float *G, const float *U, float *A, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		float g = G[i];
		A[i] = (g / (1.0f + expf(-g))) * U[i];
	}
}

// Scatter layer `layer`'s roped K and plain V into the page-block buffer
// (fp16), in the layout from kmodel.h. kvdim = n_kv_heads*head_dim.
__global__ void
k_store_kv(const float *K, const float *V, __half *blocks,
	   int layer, int L, int ntok, int P, int kvdim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ntok * kvdim)
		return;
	int e = idx % kvdim, t = idx / kvdim;
	int page = t / P, tip = t % P;
	size_t page_elems = (size_t)2 * L * P * kvdim;
	size_t kbase = (size_t)page * page_elems + ((size_t)(layer * 2 + 0) * P + tip) * kvdim + e;
	size_t vbase = (size_t)page * page_elems + ((size_t)(layer * 2 + 1) * P + tip) * kvdim + e;
	blocks[kbase] = __float2half(K[(size_t)t * kvdim + e]);
	blocks[vbase] = __float2half(V[(size_t)t * kvdim + e]);
}

// Read layer `layer`'s cached prefix K/V (fp16 page blocks) into dense fp32
// Kf/Vf[0:prefix_len][kvdim]. This is the "load cached KV" path.
__global__ void
k_unpage_prefix(const __half *blocks, int layer, int L, int prefix_len,
		int P, int kvdim, float *Kf, float *Vf)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= prefix_len * kvdim)
		return;
	int e = idx % kvdim, tp = idx / kvdim;
	int page = tp / P, tip = tp % P;
	size_t page_elems = (size_t)2 * L * P * kvdim;
	size_t kb = (size_t)page * page_elems + ((size_t)(layer * 2 + 0) * P + tip) * kvdim + e;
	size_t vb = (size_t)page * page_elems + ((size_t)(layer * 2 + 1) * P + tip) * kvdim + e;
	Kf[(size_t)tp * kvdim + e] = __half2float(blocks[kb]);
	Vf[(size_t)tp * kvdim + e] = __half2float(blocks[vb]);
}

__global__ void
k_copy(float *dst, const float *src, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		dst[i] = src[i];
}

// Attention for suffix query rows only: suffix query j is at absolute position
// prefix_len+j and attends causally over the full K/V context Kf/Vf[0..pos].
__global__ void
k_attn_cont(const float *Q, const float *Kf, const float *Vf, float *O,
	    int n_suf, int prefix_len, int nh, int nkv, int hd)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n_suf * nh)
		return;
	int h = idx % nh, j = idx / nh;
	int t = prefix_len + j;  /* absolute query position */
	int group = nh / nkv, kvh = h / group;
	float scale = 1.0f / sqrtf((float)hd);
	const float *qp = Q + (size_t)(j * nh + h) * hd;
	float sc[MAXSEQ];
	float maxv = -1e30f;
	for (int tp = 0; tp <= t; tp++) {
		const float *kp = Kf + (size_t)(tp * nkv + kvh) * hd;
		float d = 0.0f;
		for (int i = 0; i < hd; i++)
			d += qp[i] * kp[i];
		d *= scale;
		sc[tp] = d;
		if (d > maxv)
			maxv = d;
	}
	float sum = 0.0f;
	for (int tp = 0; tp <= t; tp++) {
		sc[tp] = expf(sc[tp] - maxv);
		sum += sc[tp];
	}
	float *op = O + (size_t)(j * nh + h) * hd;
	for (int i = 0; i < hd; i++)
		op[i] = 0.0f;
	for (int tp = 0; tp <= t; tp++) {
		const float *vp = Vf + (size_t)(tp * nkv + kvh) * hd;
		float w = sc[tp] / sum;
		for (int i = 0; i < hd; i++)
			op[i] += w * vp[i];
	}
}

// ---- forward --------------------------------------------------------------

void
hip_forward_logits(HipModel *hm, const uint32_t *tokens, int ntok,
		   float *logits_last_host, uint16_t *kv_blocks_host)
{
	const KCfg &c = hm->cfg;
	int H = c.hidden, hd = c.head_dim, nh = c.n_heads, nkv = c.n_kv_heads;
	int KV = nkv * hd, I = c.intermediate, V = c.vocab, P = c.page_tokens;

	uint32_t *d_tok;
	float *x, *xn, *q, *k, *v, *attn, *tmp, *gate, *up, *act, *logits;
	HIP_CHECK(hipMalloc(&d_tok, ntok * sizeof(uint32_t)));
	HIP_CHECK(hipMemcpy(d_tok, tokens, ntok * sizeof(uint32_t), hipMemcpyHostToDevice));
	HIP_CHECK(hipMalloc(&x,    (size_t)ntok * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&xn,   (size_t)ntok * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&q,    (size_t)ntok * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&k,    (size_t)ntok * KV * sizeof(float)));
	HIP_CHECK(hipMalloc(&v,    (size_t)ntok * KV * sizeof(float)));
	HIP_CHECK(hipMalloc(&attn, (size_t)ntok * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&tmp,  (size_t)ntok * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&gate, (size_t)ntok * I * sizeof(float)));
	HIP_CHECK(hipMalloc(&up,   (size_t)ntok * I * sizeof(float)));
	HIP_CHECK(hipMalloc(&act,  (size_t)ntok * I * sizeof(float)));
	HIP_CHECK(hipMalloc(&logits, (size_t)ntok * V * sizeof(float)));

	__half *d_blocks = nullptr;
	int npages = kv_num_pages(ntok, P);
	size_t block_elems = (size_t)2 * c.n_layers * P * KV;  /* per page */
	if (kv_blocks_host)
		HIP_CHECK(hipMalloc(&d_blocks, (size_t)npages * block_elems * sizeof(__half)));

	k_embed<<<NBLK(ntok * H), BLK>>>(hm->tok_emb, d_tok, x, ntok, H);

	int li = 0;
	for (auto &D : hm->layers) {
		k_rmsnorm<<<NBLK(ntok), BLK>>>(x, D.attn_norm, xn, ntok, H, c.rms_eps);
		k_mm<<<NBLK(ntok * H),  BLK>>>(xn, D.wq, q, ntok, H, H);
		k_mm<<<NBLK(ntok * KV), BLK>>>(xn, D.wk, k, ntok, H, KV);
		k_mm<<<NBLK(ntok * KV), BLK>>>(xn, D.wv, v, ntok, H, KV);
		k_rope<<<NBLK(ntok * nh  * (hd / 2)), BLK>>>(q, ntok, nh,  hd, c.rope_theta, 0);
		k_rope<<<NBLK(ntok * nkv * (hd / 2)), BLK>>>(k, ntok, nkv, hd, c.rope_theta, 0);
		if (d_blocks)  /* cache roped K + plain V in block layout */
			k_store_kv<<<NBLK(ntok * KV), BLK>>>(k, v, d_blocks, li,
							     c.n_layers, ntok, P, KV);
		k_attn<<<NBLK(ntok * nh), BLK>>>(q, k, v, attn, ntok, nh, nkv, hd);
		k_mm<<<NBLK(ntok * H), BLK>>>(attn, D.wo, tmp, ntok, H, H);
		k_add<<<NBLK(ntok * H), BLK>>>(x, tmp, ntok * H);

		k_rmsnorm<<<NBLK(ntok), BLK>>>(x, D.ffn_norm, xn, ntok, H, c.rms_eps);
		k_mm<<<NBLK(ntok * I), BLK>>>(xn, D.wg, gate, ntok, H, I);
		k_mm<<<NBLK(ntok * I), BLK>>>(xn, D.wu, up,   ntok, H, I);
		k_silumul<<<NBLK(ntok * I), BLK>>>(gate, up, act, ntok * I);
		k_mm<<<NBLK(ntok * H), BLK>>>(act, D.wd, tmp, ntok, I, H);
		k_add<<<NBLK(ntok * H), BLK>>>(x, tmp, ntok * H);
		li++;
	}

	k_rmsnorm<<<NBLK(ntok), BLK>>>(x, hm->final_norm, xn, ntok, H, c.rms_eps);
	k_mm<<<NBLK(ntok * V), BLK>>>(xn, hm->lm_head, logits, ntok, H, V);
	HIP_CHECK(hipDeviceSynchronize());

	int last = ntok - 1;
	HIP_CHECK(hipMemcpy(logits_last_host, logits + (size_t)last * V,
			    V * sizeof(float), hipMemcpyDeviceToHost));

	if (kv_blocks_host) {
		/* __half and uint16_t are both 2 bytes; copy raw bits. */
		HIP_CHECK(hipMemcpy(kv_blocks_host, d_blocks,
				    (size_t)npages * block_elems * sizeof(__half),
				    hipMemcpyDeviceToHost));
		dfree(d_blocks);
	}

	dfree(d_tok); dfree(x); dfree(xn); dfree(q); dfree(k);
	dfree(v); dfree(attn); dfree(tmp); dfree(gate); dfree(up);
	dfree(act); dfree(logits);
}

// Continuation forward: compute ONLY the suffix tokens [prefix_len, ntok),
// using the cached fp16 prefix KV blocks as attention context. Returns logits
// for the last position. prefix_len must be a multiple of page_tokens (only
// whole cached pages are reusable). This is the path that makes cached KV save
// compute; the cache-invariant test checks it matches a full recompute.
void
hip_forward_cont(HipModel *hm, const uint32_t *tokens, int ntok, int prefix_len,
		 const uint16_t *prefix_blocks_host, float *logits_last_host)
{
	const KCfg &c = hm->cfg;
	int H = c.hidden, hd = c.head_dim, nh = c.n_heads, nkv = c.n_kv_heads;
	int KV = nkv * hd, I = c.intermediate, V = c.vocab, P = c.page_tokens;
	int n_suf = ntok - prefix_len;
	int npp = prefix_len / P;                 /* cached prefix pages */
	size_t page_elems = (size_t)2 * c.n_layers * P * KV;

	uint32_t *d_suftok;
	__half *d_pref = nullptr;
	float *xs, *xn, *qs, *ks, *vs, *attn, *tmp, *gate, *up, *act, *logits;
	float *Kf, *Vf;
	HIP_CHECK(hipMalloc(&d_suftok, n_suf * sizeof(uint32_t)));
	HIP_CHECK(hipMemcpy(d_suftok, tokens + prefix_len, n_suf * sizeof(uint32_t),
			    hipMemcpyHostToDevice));
	if (npp > 0) {
		HIP_CHECK(hipMalloc(&d_pref, (size_t)npp * page_elems * sizeof(__half)));
		HIP_CHECK(hipMemcpy(d_pref, prefix_blocks_host,
				    (size_t)npp * page_elems * sizeof(__half),
				    hipMemcpyHostToDevice));
	}
	HIP_CHECK(hipMalloc(&xs,   (size_t)n_suf * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&xn,   (size_t)n_suf * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&qs,   (size_t)n_suf * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&ks,   (size_t)n_suf * KV * sizeof(float)));
	HIP_CHECK(hipMalloc(&vs,   (size_t)n_suf * KV * sizeof(float)));
	HIP_CHECK(hipMalloc(&attn, (size_t)n_suf * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&tmp,  (size_t)n_suf * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&gate, (size_t)n_suf * I * sizeof(float)));
	HIP_CHECK(hipMalloc(&up,   (size_t)n_suf * I * sizeof(float)));
	HIP_CHECK(hipMalloc(&act,  (size_t)n_suf * I * sizeof(float)));
	HIP_CHECK(hipMalloc(&logits, (size_t)n_suf * V * sizeof(float)));
	HIP_CHECK(hipMalloc(&Kf, (size_t)ntok * KV * sizeof(float)));  /* full ctx */
	HIP_CHECK(hipMalloc(&Vf, (size_t)ntok * KV * sizeof(float)));

	k_embed<<<NBLK(n_suf * H), BLK>>>(hm->tok_emb, d_suftok, xs, n_suf, H);

	int li = 0;
	for (auto &D : hm->layers) {
		k_rmsnorm<<<NBLK(n_suf), BLK>>>(xs, D.attn_norm, xn, n_suf, H, c.rms_eps);
		k_mm<<<NBLK(n_suf * H),  BLK>>>(xn, D.wq, qs, n_suf, H, H);
		k_mm<<<NBLK(n_suf * KV), BLK>>>(xn, D.wk, ks, n_suf, H, KV);
		k_mm<<<NBLK(n_suf * KV), BLK>>>(xn, D.wv, vs, n_suf, H, KV);
		k_rope<<<NBLK(n_suf * nh  * (hd / 2)), BLK>>>(qs, n_suf, nh,  hd, c.rope_theta, prefix_len);
		k_rope<<<NBLK(n_suf * nkv * (hd / 2)), BLK>>>(ks, n_suf, nkv, hd, c.rope_theta, prefix_len);

		/* build full K/V context: cached prefix (fp16) + fresh suffix */
		if (npp > 0)
			k_unpage_prefix<<<NBLK(prefix_len * KV), BLK>>>(
				d_pref, li, c.n_layers, prefix_len, P, KV, Kf, Vf);
		k_copy<<<NBLK(n_suf * KV), BLK>>>(Kf + (size_t)prefix_len * KV, ks, n_suf * KV);
		k_copy<<<NBLK(n_suf * KV), BLK>>>(Vf + (size_t)prefix_len * KV, vs, n_suf * KV);

		k_attn_cont<<<NBLK(n_suf * nh), BLK>>>(qs, Kf, Vf, attn, n_suf, prefix_len, nh, nkv, hd);
		k_mm<<<NBLK(n_suf * H), BLK>>>(attn, D.wo, tmp, n_suf, H, H);
		k_add<<<NBLK(n_suf * H), BLK>>>(xs, tmp, n_suf * H);

		k_rmsnorm<<<NBLK(n_suf), BLK>>>(xs, D.ffn_norm, xn, n_suf, H, c.rms_eps);
		k_mm<<<NBLK(n_suf * I), BLK>>>(xn, D.wg, gate, n_suf, H, I);
		k_mm<<<NBLK(n_suf * I), BLK>>>(xn, D.wu, up,   n_suf, H, I);
		k_silumul<<<NBLK(n_suf * I), BLK>>>(gate, up, act, n_suf * I);
		k_mm<<<NBLK(n_suf * H), BLK>>>(act, D.wd, tmp, n_suf, I, H);
		k_add<<<NBLK(n_suf * H), BLK>>>(xs, tmp, n_suf * H);
		li++;
	}

	k_rmsnorm<<<NBLK(n_suf), BLK>>>(xs, hm->final_norm, xn, n_suf, H, c.rms_eps);
	k_mm<<<NBLK(n_suf * V), BLK>>>(xn, hm->lm_head, logits, n_suf, H, V);
	HIP_CHECK(hipDeviceSynchronize());

	HIP_CHECK(hipMemcpy(logits_last_host, logits + (size_t)(n_suf - 1) * V,
			    V * sizeof(float), hipMemcpyDeviceToHost));

	dfree(d_suftok); dfree(d_pref); dfree(xs); dfree(xn); dfree(qs);
	dfree(ks); dfree(vs); dfree(attn); dfree(tmp); dfree(gate);
	dfree(up); dfree(act); dfree(logits); dfree(Kf); dfree(Vf);
}
