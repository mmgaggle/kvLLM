// SPDX-License-Identifier: BSD-3-Clause
// IBM Granite 4.0 dense transformer in HIP (fp32).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "granite.h"

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

struct GLayer {
	float *in_ln, *wq, *wk, *wv, *wo, *post_ln, *w_in, *w_out;
};

struct Granite {
	int L, H, NH, NKV, HD, I, V, KV;   /* KV = NKV*HD */
	float rope_theta, rms_eps, attn_mult, embed_mult, resid_mult, logits_scale;
	float *embed, *final_norm;
	std::vector<GLayer> layers;
};

static void dfree(void *p) { if (p) (void)hipFree(p); }

// ---- kernels ---------------------------------------------------------------

__global__ void
k_embed(const float *embed, const uint32_t *tok, float *x, int ntok, int H, float mult)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ntok * H) return;
	int t = idx / H, i = idx % H;
	x[idx] = embed[(size_t)tok[t] * H + i] * mult;
}

__global__ void
k_rmsnorm(const float *X, const float *w, float *Y, int M, int H, float eps)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= M) return;
	const float *x = X + (size_t)t * H;
	float ss = 0.0f;
	for (int i = 0; i < H; i++) ss += x[i] * x[i];
	float inv = 1.0f / sqrtf(ss / H + eps);
	float *y = Y + (size_t)t * H;
	for (int i = 0; i < H; i++) y[i] = x[i] * inv * w[i];
}

// Y[M][N] = X[M][K] @ W[K][N]
__global__ void
k_mm(const float *X, const float *W, float *Y, int M, int K, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= M * N) return;
	int m = idx / N, n = idx % N;
	float a = 0.0f;
	for (int k = 0; k < K; k++) a += X[m * K + k] * W[k * N + n];
	Y[idx] = a;
}

// rotate-half RoPE (HF convention): pairs (j, j+hd/2). pos0 = abs position of row 0.
__global__ void
k_rope(float *B, int M, int nh, int hd, float theta, int pos0)
{
	int half = hd / 2;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= M * nh * half) return;
	int j = idx % half;
	int h = (idx / half) % nh;
	int t = idx / half / nh;
	float inv = powf(theta, -2.0f * j / hd);
	float ang = (pos0 + t) * inv;
	float c = cosf(ang), s = sinf(ang);
	float *base = B + (size_t)(t * nh + h) * hd;
	float x0 = base[j], x1 = base[j + half];
	base[j]        = x0 * c - x1 * s;
	base[j + half] = x1 * c + x0 * s;
}

// Causal GQA attention (separate q/k/v buffers). Online softmax. scale applied to scores.
__global__ void
k_attn(const float *Q, const float *K, const float *V, float *O,
       int M, int nh, int nkv, int hd, int kvdim, float scale)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= M * nh) return;
	int h = idx % nh, t = idx / nh;
	int group = nh / nkv, kvh = h / group;
	const float *q = Q + (size_t)t * nh * hd + h * hd;
	float acc[128];
	for (int i = 0; i < hd; i++) acc[i] = 0.0f;
	float m = -1e30f, l = 0.0f;
	for (int tp = 0; tp <= t; tp++) {
		const float *k = K + (size_t)tp * kvdim + kvh * hd;
		float s = 0.0f;
		for (int i = 0; i < hd; i++) s += q[i] * k[i];
		s *= scale;
		float mn = fmaxf(m, s), corr = expf(m - mn), p = expf(s - mn);
		l = l * corr + p;
		const float *v = V + (size_t)tp * kvdim + kvh * hd;
		for (int i = 0; i < hd; i++) acc[i] = acc[i] * corr + p * v[i];
		m = mn;
	}
	float *o = O + (size_t)t * nh * hd + h * hd;
	for (int i = 0; i < hd; i++) o[i] = acc[i] / l;
}

__global__ void
k_residual(float *X, const float *sub, float mult, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) X[i] += mult * sub[i];
}

// SwiGLU with fused gate|up: GU is [M][2I] (gate = [0,I), up = [I,2I)).
__global__ void
k_silu_glu(const float *GU, float *A, int M, int I)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= M * I) return;
	int m = idx / I, i = idx % I;
	const float *gu = GU + (size_t)m * 2 * I;
	float g = gu[i];
	A[idx] = (g / (1.0f + expf(-g))) * gu[I + i];
}

// Tied lm_head: logits[v] = (x_last . embed[v]) * inv_scale   (embed is [V][H]).
__global__ void
k_lm_head(const float *x_last, const float *embed, float *logits, int H, int V, float inv_scale)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	if (v >= V) return;
	const float *w = embed + (size_t)v * H;
	float a = 0.0f;
	for (int i = 0; i < H; i++) a += x_last[i] * w[i];
	logits[v] = a * inv_scale;
}

// Scatter K/V (kvdim each) into fp16 page blocks: ((layer*2+kv)*P + tip)*kvdim + e.
__global__ void
k_store_kv(const float *K, const float *V, __half *blocks, int layer, int L,
	   int ntok, int P, int kvdim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ntok * kvdim) return;
	int e = idx % kvdim, t = idx / kvdim;
	int page = t / P, tip = t % P;
	size_t pe = (size_t)2 * L * P * kvdim;
	size_t kb = (size_t)page * pe + ((size_t)(layer * 2 + 0) * P + tip) * kvdim + e;
	size_t vb = (size_t)page * pe + ((size_t)(layer * 2 + 1) * P + tip) * kvdim + e;
	blocks[kb] = __float2half(K[(size_t)t * kvdim + e]);
	blocks[vb] = __float2half(V[(size_t)t * kvdim + e]);
}

// ---- load ------------------------------------------------------------------

static float *
read_dev(FILE *f, size_t n)
{
	std::vector<float> host(n);
	if (fread(host.data(), sizeof(float), n, f) != n) {
		fprintf(stderr, "granite_load: short read\n"); exit(1);
	}
	float *d = nullptr;
	HIP_CHECK(hipMalloc(&d, n * sizeof(float)));
	HIP_CHECK(hipMemcpy(d, host.data(), n * sizeof(float), hipMemcpyHostToDevice));
	return d;
}

Granite *
granite_load(const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) { fprintf(stderr, "granite_load: cannot open %s\n", path); return nullptr; }
	int32_t hd[8];
	float fp[6];
	if (fread(hd, 4, 8, f) != 8 || hd[0] != 0x4752414E ||
	    fread(fp, 4, 6, f) != 6) {
		fprintf(stderr, "granite_load: bad header\n"); fclose(f); return nullptr;
	}
	auto *g = new Granite();
	g->L = hd[1]; g->H = hd[2]; g->NH = hd[3]; g->NKV = hd[4];
	g->HD = hd[5]; g->I = hd[6]; g->V = hd[7]; g->KV = g->NKV * g->HD;
	g->rope_theta = fp[0]; g->rms_eps = fp[1]; g->attn_mult = fp[2];
	g->embed_mult = fp[3]; g->resid_mult = fp[4]; g->logits_scale = fp[5];
	int H = g->H, KV = g->KV, I = g->I;

	g->embed = read_dev(f, (size_t)g->V * H);
	g->final_norm = read_dev(f, H);
	g->layers.resize(g->L);
	for (auto &L : g->layers) {
		L.in_ln   = read_dev(f, H);
		L.wq      = read_dev(f, (size_t)H * H);
		L.wk      = read_dev(f, (size_t)H * KV);
		L.wv      = read_dev(f, (size_t)H * KV);
		L.wo      = read_dev(f, (size_t)H * H);
		L.post_ln = read_dev(f, H);
		L.w_in    = read_dev(f, (size_t)H * 2 * I);
		L.w_out   = read_dev(f, (size_t)I * H);
	}
	fclose(f);
	return g;
}

void
granite_free(Granite *g)
{
	if (!g) return;
	dfree(g->embed); dfree(g->final_norm);
	for (auto &L : g->layers) {
		dfree(L.in_ln); dfree(L.wq); dfree(L.wk); dfree(L.wv); dfree(L.wo);
		dfree(L.post_ln); dfree(L.w_in); dfree(L.w_out);
	}
	delete g;
}

int granite_vocab(const Granite *g) { return g->V; }

size_t
granite_kv_block_elems(const Granite *g, int page_tokens)
{
	return (size_t)2 * g->L * page_tokens * g->KV;
}

// ---- batched forward (prefill / cold) --------------------------------------

void
granite_forward_logits(Granite *g, const uint32_t *toks, int ntok, float *logits_host,
		       uint16_t *kv_blocks_host, int page_tokens)
{
	int H = g->H, NH = g->NH, NKV = g->NKV, HD = g->HD, KV = g->KV, I = g->I, V = g->V, P = page_tokens;

	uint32_t *d_tok;
	float *x, *xn, *q, *k, *v, *attn, *tmp, *gu, *act, *logits;
	HIP_CHECK(hipMalloc(&d_tok, ntok * sizeof(uint32_t)));
	HIP_CHECK(hipMemcpy(d_tok, toks, ntok * sizeof(uint32_t), hipMemcpyHostToDevice));
	HIP_CHECK(hipMalloc(&x,    (size_t)ntok * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&xn,   (size_t)ntok * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&q,    (size_t)ntok * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&k,    (size_t)ntok * KV * sizeof(float)));
	HIP_CHECK(hipMalloc(&v,    (size_t)ntok * KV * sizeof(float)));
	HIP_CHECK(hipMalloc(&attn, (size_t)ntok * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&tmp,  (size_t)ntok * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&gu,   (size_t)ntok * 2 * I * sizeof(float)));
	HIP_CHECK(hipMalloc(&act,  (size_t)ntok * I * sizeof(float)));
	HIP_CHECK(hipMalloc(&logits, (size_t)V * sizeof(float)));

	__half *d_blocks = nullptr;
	int npages = (ntok + P - 1) / P;
	size_t block_elems = (size_t)2 * g->L * P * KV;
	if (kv_blocks_host)
		HIP_CHECK(hipMalloc(&d_blocks, (size_t)npages * block_elems * sizeof(__half)));

	k_embed<<<NBLK(ntok * H), BLK>>>(g->embed, d_tok, x, ntok, H, g->embed_mult);

	int li = 0;
	for (auto &L : g->layers) {
		k_rmsnorm<<<NBLK(ntok), BLK>>>(x, L.in_ln, xn, ntok, H, g->rms_eps);
		k_mm<<<NBLK(ntok * H),  BLK>>>(xn, L.wq, q, ntok, H, H);
		k_mm<<<NBLK(ntok * KV), BLK>>>(xn, L.wk, k, ntok, H, KV);
		k_mm<<<NBLK(ntok * KV), BLK>>>(xn, L.wv, v, ntok, H, KV);
		k_rope<<<NBLK(ntok * NH  * (HD / 2)), BLK>>>(q, ntok, NH,  HD, g->rope_theta, 0);
		k_rope<<<NBLK(ntok * NKV * (HD / 2)), BLK>>>(k, ntok, NKV, HD, g->rope_theta, 0);
		if (d_blocks)
			k_store_kv<<<NBLK(ntok * KV), BLK>>>(k, v, d_blocks, li, g->L, ntok, P, KV);
		k_attn<<<NBLK(ntok * NH), BLK>>>(q, k, v, attn, ntok, NH, NKV, HD, KV, g->attn_mult);
		k_mm<<<NBLK(ntok * H), BLK>>>(attn, L.wo, tmp, ntok, H, H);
		k_residual<<<NBLK(ntok * H), BLK>>>(x, tmp, g->resid_mult, ntok * H);

		k_rmsnorm<<<NBLK(ntok), BLK>>>(x, L.post_ln, xn, ntok, H, g->rms_eps);
		k_mm<<<NBLK(ntok * 2 * I), BLK>>>(xn, L.w_in, gu, ntok, H, 2 * I);
		k_silu_glu<<<NBLK(ntok * I), BLK>>>(gu, act, ntok, I);
		k_mm<<<NBLK(ntok * H), BLK>>>(act, L.w_out, tmp, ntok, I, H);
		k_residual<<<NBLK(ntok * H), BLK>>>(x, tmp, g->resid_mult, ntok * H);
		li++;
	}

	k_rmsnorm<<<NBLK(ntok), BLK>>>(x, g->final_norm, xn, ntok, H, g->rms_eps);
	k_lm_head<<<NBLK(V), BLK>>>(xn + (size_t)(ntok - 1) * H, g->embed, logits, H, V, 1.0f / g->logits_scale);
	HIP_CHECK(hipDeviceSynchronize());

	HIP_CHECK(hipMemcpy(logits_host, logits, V * sizeof(float), hipMemcpyDeviceToHost));
	if (kv_blocks_host) {
		HIP_CHECK(hipMemcpy(kv_blocks_host, d_blocks,
				    (size_t)npages * block_elems * sizeof(__half), hipMemcpyDeviceToHost));
		dfree(d_blocks);
	}
	dfree(d_tok); dfree(x); dfree(xn); dfree(q); dfree(k); dfree(v);
	dfree(attn); dfree(tmp); dfree(gu); dfree(act); dfree(logits);
}

void
granite_generate(Granite *g, std::vector<uint32_t> &toks, int n_new, std::vector<uint32_t> &out)
{
	std::vector<float> logits(g->V);
	for (int s = 0; s < n_new; s++) {
		granite_forward_logits(g, toks.data(), (int)toks.size(), logits.data());
		int best = 0;
		for (int vv = 1; vv < g->V; vv++) if (logits[vv] > logits[best]) best = vv;
		toks.push_back((uint32_t)best);
		out.push_back((uint32_t)best);
	}
}

// ---- incremental KV cache --------------------------------------------------

struct GraniteKV {
	int maxctx, len;
	std::vector<float *> dK, dV;            /* per layer [maxctx*KV] */
	uint32_t *d_tok1;
	float *x1, *xn1, *q1, *k1, *v1, *attn1, *tmp1, *gu1, *act1, *logits;
};

GraniteKV *
granite_kv_create(Granite *g, int maxctx)
{
	int H = g->H, KV = g->KV, I = g->I;
	auto *kv = new GraniteKV();
	kv->maxctx = maxctx; kv->len = 0;
	kv->dK.resize(g->L); kv->dV.resize(g->L);
	for (int l = 0; l < g->L; l++) {
		HIP_CHECK(hipMalloc(&kv->dK[l], (size_t)maxctx * KV * sizeof(float)));
		HIP_CHECK(hipMalloc(&kv->dV[l], (size_t)maxctx * KV * sizeof(float)));
	}
	HIP_CHECK(hipMalloc(&kv->d_tok1, sizeof(uint32_t)));
	HIP_CHECK(hipMalloc(&kv->x1,   (size_t)H * sizeof(float)));
	HIP_CHECK(hipMalloc(&kv->xn1,  (size_t)H * sizeof(float)));
	HIP_CHECK(hipMalloc(&kv->q1,   (size_t)H * sizeof(float)));
	HIP_CHECK(hipMalloc(&kv->k1,   (size_t)KV * sizeof(float)));
	HIP_CHECK(hipMalloc(&kv->v1,   (size_t)KV * sizeof(float)));
	HIP_CHECK(hipMalloc(&kv->attn1,(size_t)H * sizeof(float)));
	HIP_CHECK(hipMalloc(&kv->tmp1, (size_t)H * sizeof(float)));
	HIP_CHECK(hipMalloc(&kv->gu1,  (size_t)2 * I * sizeof(float)));
	HIP_CHECK(hipMalloc(&kv->act1, (size_t)I * sizeof(float)));
	HIP_CHECK(hipMalloc(&kv->logits, (size_t)g->V * sizeof(float)));
	return kv;
}

void
granite_kv_free(GraniteKV *kv)
{
	if (!kv) return;
	for (auto p : kv->dK) dfree(p);
	for (auto p : kv->dV) dfree(p);
	dfree(kv->d_tok1); dfree(kv->x1); dfree(kv->xn1); dfree(kv->q1); dfree(kv->k1);
	dfree(kv->v1); dfree(kv->attn1); dfree(kv->tmp1); dfree(kv->gu1); dfree(kv->act1); dfree(kv->logits);
	delete kv;
}

__global__ void
k_append_kv(const float *K, const float *V, float *dK, float *dV, int base, int ntok, int kvdim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ntok * kvdim) return;
	int e = idx % kvdim, t = idx / kvdim;
	dK[(size_t)(base + t) * kvdim + e] = K[(size_t)t * kvdim + e];
	dV[(size_t)(base + t) * kvdim + e] = V[(size_t)t * kvdim + e];
}

// Single-query GQA decode attention over the cache [0,curlen).
__global__ void
k_attn_decode(const float *q, const float *dK, const float *dV, float *out,
	      int curlen, int nh, int nkv, int hd, int kvdim, float scale)
{
	int h = blockIdx.x * blockDim.x + threadIdx.x;
	if (h >= nh) return;
	int group = nh / nkv, kvh = h / group;
	const float *qh = q + h * hd;
	float acc[128];
	for (int i = 0; i < hd; i++) acc[i] = 0.0f;
	float m = -1e30f, l = 0.0f;
	for (int tp = 0; tp < curlen; tp++) {
		const float *k = dK + (size_t)tp * kvdim + kvh * hd;
		float s = 0.0f;
		for (int i = 0; i < hd; i++) s += qh[i] * k[i];
		s *= scale;
		float mn = fmaxf(m, s), corr = expf(m - mn), p = expf(s - mn);
		l = l * corr + p;
		const float *v = dV + (size_t)tp * kvdim + kvh * hd;
		for (int i = 0; i < hd; i++) acc[i] = acc[i] * corr + p * v[i];
		m = mn;
	}
	float *o = out + h * hd;
	for (int i = 0; i < hd; i++) o[i] = acc[i] / l;
}

static void
granite_prefill(Granite *g, GraniteKV *kv, const uint32_t *toks, int ntok, float *logits_host)
{
	int H = g->H, NH = g->NH, NKV = g->NKV, HD = g->HD, KV = g->KV, I = g->I, V = g->V;
	uint32_t *d_tok;
	float *x, *xn, *q, *k, *v, *attn, *tmp, *gu, *act, *logits;
	HIP_CHECK(hipMalloc(&d_tok, ntok * sizeof(uint32_t)));
	HIP_CHECK(hipMemcpy(d_tok, toks, ntok * sizeof(uint32_t), hipMemcpyHostToDevice));
	HIP_CHECK(hipMalloc(&x,    (size_t)ntok * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&xn,   (size_t)ntok * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&q,    (size_t)ntok * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&k,    (size_t)ntok * KV * sizeof(float)));
	HIP_CHECK(hipMalloc(&v,    (size_t)ntok * KV * sizeof(float)));
	HIP_CHECK(hipMalloc(&attn, (size_t)ntok * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&tmp,  (size_t)ntok * H * sizeof(float)));
	HIP_CHECK(hipMalloc(&gu,   (size_t)ntok * 2 * I * sizeof(float)));
	HIP_CHECK(hipMalloc(&act,  (size_t)ntok * I * sizeof(float)));
	HIP_CHECK(hipMalloc(&logits, (size_t)V * sizeof(float)));

	k_embed<<<NBLK(ntok * H), BLK>>>(g->embed, d_tok, x, ntok, H, g->embed_mult);
	int li = 0;
	for (auto &L : g->layers) {
		k_rmsnorm<<<NBLK(ntok), BLK>>>(x, L.in_ln, xn, ntok, H, g->rms_eps);
		k_mm<<<NBLK(ntok * H),  BLK>>>(xn, L.wq, q, ntok, H, H);
		k_mm<<<NBLK(ntok * KV), BLK>>>(xn, L.wk, k, ntok, H, KV);
		k_mm<<<NBLK(ntok * KV), BLK>>>(xn, L.wv, v, ntok, H, KV);
		k_rope<<<NBLK(ntok * NH  * (HD / 2)), BLK>>>(q, ntok, NH,  HD, g->rope_theta, 0);
		k_rope<<<NBLK(ntok * NKV * (HD / 2)), BLK>>>(k, ntok, NKV, HD, g->rope_theta, 0);
		k_append_kv<<<NBLK(ntok * KV), BLK>>>(k, v, kv->dK[li], kv->dV[li], 0, ntok, KV);
		k_attn<<<NBLK(ntok * NH), BLK>>>(q, k, v, attn, ntok, NH, NKV, HD, KV, g->attn_mult);
		k_mm<<<NBLK(ntok * H), BLK>>>(attn, L.wo, tmp, ntok, H, H);
		k_residual<<<NBLK(ntok * H), BLK>>>(x, tmp, g->resid_mult, ntok * H);
		k_rmsnorm<<<NBLK(ntok), BLK>>>(x, L.post_ln, xn, ntok, H, g->rms_eps);
		k_mm<<<NBLK(ntok * 2 * I), BLK>>>(xn, L.w_in, gu, ntok, H, 2 * I);
		k_silu_glu<<<NBLK(ntok * I), BLK>>>(gu, act, ntok, I);
		k_mm<<<NBLK(ntok * H), BLK>>>(act, L.w_out, tmp, ntok, I, H);
		k_residual<<<NBLK(ntok * H), BLK>>>(x, tmp, g->resid_mult, ntok * H);
		li++;
	}
	k_rmsnorm<<<NBLK(ntok), BLK>>>(x, g->final_norm, xn, ntok, H, g->rms_eps);
	k_lm_head<<<NBLK(V), BLK>>>(xn + (size_t)(ntok - 1) * H, g->embed, logits, H, V, 1.0f / g->logits_scale);
	HIP_CHECK(hipDeviceSynchronize());
	HIP_CHECK(hipMemcpy(logits_host, logits, V * sizeof(float), hipMemcpyDeviceToHost));
	kv->len = ntok;
	dfree(d_tok); dfree(x); dfree(xn); dfree(q); dfree(k); dfree(v);
	dfree(attn); dfree(tmp); dfree(gu); dfree(act); dfree(logits);
}

static void
granite_decode_step(Granite *g, GraniteKV *kv, uint32_t token, float *logits_host)
{
	int H = g->H, NH = g->NH, NKV = g->NKV, HD = g->HD, KV = g->KV, I = g->I, V = g->V;
	int pos = kv->len;
	HIP_CHECK(hipMemcpy(kv->d_tok1, &token, sizeof(uint32_t), hipMemcpyHostToDevice));
	k_embed<<<NBLK(H), BLK>>>(g->embed, kv->d_tok1, kv->x1, 1, H, g->embed_mult);
	int li = 0;
	for (auto &L : g->layers) {
		k_rmsnorm<<<1, BLK>>>(kv->x1, L.in_ln, kv->xn1, 1, H, g->rms_eps);
		k_mm<<<NBLK(H),  BLK>>>(kv->xn1, L.wq, kv->q1, 1, H, H);
		k_mm<<<NBLK(KV), BLK>>>(kv->xn1, L.wk, kv->k1, 1, H, KV);
		k_mm<<<NBLK(KV), BLK>>>(kv->xn1, L.wv, kv->v1, 1, H, KV);
		k_rope<<<NBLK(NH  * (HD / 2)), BLK>>>(kv->q1, 1, NH,  HD, g->rope_theta, pos);
		k_rope<<<NBLK(NKV * (HD / 2)), BLK>>>(kv->k1, 1, NKV, HD, g->rope_theta, pos);
		k_append_kv<<<NBLK(KV), BLK>>>(kv->k1, kv->v1, kv->dK[li], kv->dV[li], pos, 1, KV);
		k_attn_decode<<<NBLK(NH), BLK>>>(kv->q1, kv->dK[li], kv->dV[li], kv->attn1,
						 pos + 1, NH, NKV, HD, KV, g->attn_mult);
		k_mm<<<NBLK(H), BLK>>>(kv->attn1, L.wo, kv->tmp1, 1, H, H);
		k_residual<<<NBLK(H), BLK>>>(kv->x1, kv->tmp1, g->resid_mult, H);
		k_rmsnorm<<<1, BLK>>>(kv->x1, L.post_ln, kv->xn1, 1, H, g->rms_eps);
		k_mm<<<NBLK(2 * I), BLK>>>(kv->xn1, L.w_in, kv->gu1, 1, H, 2 * I);
		k_silu_glu<<<NBLK(I), BLK>>>(kv->gu1, kv->act1, 1, I);
		k_mm<<<NBLK(H), BLK>>>(kv->act1, L.w_out, kv->tmp1, 1, I, H);
		k_residual<<<NBLK(H), BLK>>>(kv->x1, kv->tmp1, g->resid_mult, H);
		li++;
	}
	k_rmsnorm<<<1, BLK>>>(kv->x1, g->final_norm, kv->xn1, 1, H, g->rms_eps);
	k_lm_head<<<NBLK(V), BLK>>>(kv->xn1, g->embed, kv->logits, H, V, 1.0f / g->logits_scale);
	HIP_CHECK(hipDeviceSynchronize());
	HIP_CHECK(hipMemcpy(logits_host, kv->logits, V * sizeof(float), hipMemcpyDeviceToHost));
	kv->len = pos + 1;
}

static uint32_t
pick_token(const std::vector<float> &logits, int V, float temp, uint64_t &rng)
{
	if (temp <= 0.0f) {
		int best = 0;
		for (int v = 1; v < V; v++) if (logits[v] > logits[best]) best = v;
		return (uint32_t)best;
	}
	float mx = logits[0];
	for (int v = 1; v < V; v++) mx = fmaxf(mx, logits[v]);
	double sum = 0.0;
	std::vector<double> p(V);
	for (int v = 0; v < V; v++) { p[v] = exp((logits[v] - mx) / temp); sum += p[v]; }
	rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
	double u = ((double)(rng >> 11) / (double)(1ULL << 53)) * sum, acc = 0.0;
	for (int v = 0; v < V; v++) { acc += p[v]; if (u < acc) return (uint32_t)v; }
	return (uint32_t)(V - 1);
}

static void
decode_loop(Granite *g, GraniteKV *kv, std::vector<uint32_t> &toks, int n_new,
	    std::vector<uint32_t> &out, std::vector<float> &logits, float temp, uint64_t seed)
{
	uint64_t rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
	for (int s = 0; s < n_new; s++) {
		uint32_t next = pick_token(logits, g->V, temp, rng);
		toks.push_back(next);
		out.push_back(next);
		granite_decode_step(g, kv, next, logits.data());
	}
}

void
granite_generate_cached(Granite *g, GraniteKV *kv, std::vector<uint32_t> &toks,
			int n_new, std::vector<uint32_t> &out, float temp, uint64_t seed)
{
	std::vector<float> logits(g->V);
	granite_prefill(g, kv, toks.data(), (int)toks.size(), logits.data());
	decode_loop(g, kv, toks, n_new, out, logits, temp, seed);
}

// ---- warm-start ------------------------------------------------------------

__global__ void
k_load_cache(const __half *blocks, float *dK, float *dV, int layer, int L,
	     int n_load, int P, int kvdim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n_load * kvdim) return;
	int e = idx % kvdim, pos = idx / kvdim;
	int page = pos / P, tip = pos % P;
	size_t pe = (size_t)2 * L * P * kvdim;
	size_t kb = (size_t)page * pe + ((size_t)(layer * 2 + 0) * P + tip) * kvdim + e;
	size_t vb = (size_t)page * pe + ((size_t)(layer * 2 + 1) * P + tip) * kvdim + e;
	dK[(size_t)pos * kvdim + e] = __half2float(blocks[kb]);
	dV[(size_t)pos * kvdim + e] = __half2float(blocks[vb]);
}

__global__ void
k_pack_blocks(const float *dK, const float *dV, __half *blocks, int layer, int L,
	      int len, int P, int kvdim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= len * kvdim) return;
	int e = idx % kvdim, pos = idx / kvdim;
	int page = pos / P, tip = pos % P;
	size_t pe = (size_t)2 * L * P * kvdim;
	size_t kb = (size_t)page * pe + ((size_t)(layer * 2 + 0) * P + tip) * kvdim + e;
	size_t vb = (size_t)page * pe + ((size_t)(layer * 2 + 1) * P + tip) * kvdim + e;
	blocks[kb] = __float2half(dK[(size_t)pos * kvdim + e]);
	blocks[vb] = __float2half(dV[(size_t)pos * kvdim + e]);
}

void
granite_kv_to_blocks(Granite *g, GraniteKV *kv, int page_tokens, uint16_t *blocks_host)
{
	int KV = g->KV, len = kv->len, P = page_tokens;
	int npages = (len + P - 1) / P;
	size_t be = (size_t)2 * g->L * P * KV;
	__half *d_blocks;
	HIP_CHECK(hipMalloc(&d_blocks, (size_t)npages * be * sizeof(__half)));
	HIP_CHECK(hipMemset(d_blocks, 0, (size_t)npages * be * sizeof(__half)));
	for (int l = 0; l < g->L; l++)
		k_pack_blocks<<<NBLK(len * KV), BLK>>>(kv->dK[l], kv->dV[l], d_blocks, l, g->L, len, P, KV);
	HIP_CHECK(hipDeviceSynchronize());
	HIP_CHECK(hipMemcpy(blocks_host, d_blocks, (size_t)npages * be * sizeof(__half), hipMemcpyDeviceToHost));
	dfree(d_blocks);
}

void
granite_generate_warm(Granite *g, GraniteKV *kv, const uint16_t *prefix_blocks,
		      int n_load, int page_tokens, std::vector<uint32_t> &toks,
		      int n_new, std::vector<uint32_t> &out, float temp, uint64_t seed)
{
	int KV = g->KV, ntok = (int)toks.size();
	int npages = (n_load + page_tokens - 1) / page_tokens;
	size_t be = (size_t)2 * g->L * page_tokens * KV;

	__half *d_blocks;
	HIP_CHECK(hipMalloc(&d_blocks, (size_t)npages * be * sizeof(__half)));
	HIP_CHECK(hipMemcpy(d_blocks, prefix_blocks, (size_t)npages * be * sizeof(__half), hipMemcpyHostToDevice));
	for (int l = 0; l < g->L; l++)
		k_load_cache<<<NBLK(n_load * KV), BLK>>>(d_blocks, kv->dK[l], kv->dV[l], l, g->L, n_load, page_tokens, KV);
	HIP_CHECK(hipDeviceSynchronize());
	dfree(d_blocks);
	kv->len = n_load;

	std::vector<float> logits(g->V);
	for (int t = n_load; t < ntok; t++)
		granite_decode_step(g, kv, toks[t], logits.data());
	decode_loop(g, kv, toks, n_new, out, logits, temp, seed);
}
