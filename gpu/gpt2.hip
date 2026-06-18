// SPDX-License-Identifier: BSD-3-Clause
// GPT-2 forward pass in HIP (fp32).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "gpt2.h"

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
#define LN_EPS 1e-5f

struct G2Layer {
	float *ln1_w, *ln1_b, *cattn_w, *cattn_b, *cproj_w, *cproj_b;
	float *ln2_w, *ln2_b, *cfc_w, *cfc_b, *mcproj_w, *mcproj_b;
};

struct Gpt2 {
	int L, E, H, P, V;
	float *wte, *wpe, *lnf_w, *lnf_b;
	std::vector<G2Layer> layers;
};

static void
dfree(void *p) { if (p) (void)hipFree(p); }

// ---- kernels ---------------------------------------------------------------

// pos0 = absolute position of row 0 (0 for prefill, the cache length for a
// single-token decode step).
__global__ void
k_embed_pos(const float *wte, const float *wpe, const uint32_t *tok,
	    float *x, int ntok, int E, int pos0)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ntok * E)
		return;
	int t = idx / E, i = idx % E;
	x[idx] = wte[(size_t)tok[t] * E + i] + wpe[(size_t)(pos0 + t) * E + i];
}

// Append ntok tokens' K and V (from the combined QKV buffer) into the flat
// per-layer KV cache at positions [base, base+ntok).
__global__ void
k_append_kv(const float *QKV, float *dK, float *dV, int base, int ntok, int E)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ntok * E)
		return;
	int e = idx % E, t = idx / E;
	dK[(size_t)(base + t) * E + e] = QKV[(size_t)t * 3 * E + E + e];
	dV[(size_t)(base + t) * E + e] = QKV[(size_t)t * 3 * E + 2 * E + e];
}

// Single-query decode attention: the new token's q (in qkv[0]) attends over the
// cached K/V [0, curlen). Online softmax. One thread per head.
__global__ void
k_attn_decode(const float *qkv, const float *dK, const float *dV, float *out,
	      int curlen, int nh, int hd, int E)
{
	int h = blockIdx.x * blockDim.x + threadIdx.x;
	if (h >= nh)
		return;
	const float *q = qkv + h * hd;
	float scale = 1.0f / sqrtf((float)hd);
	float acc[128];
	for (int i = 0; i < hd; i++)
		acc[i] = 0.0f;
	float m = -1e30f, l = 0.0f;
	for (int tp = 0; tp < curlen; tp++) {
		const float *k = dK + (size_t)tp * E + h * hd;
		float s = 0.0f;
		for (int i = 0; i < hd; i++)
			s += q[i] * k[i];
		s *= scale;
		float mn = fmaxf(m, s), corr = expf(m - mn), p = expf(s - mn);
		l = l * corr + p;
		const float *v = dV + (size_t)tp * E + h * hd;
		for (int i = 0; i < hd; i++)
			acc[i] = acc[i] * corr + p * v[i];
		m = mn;
	}
	float *o = out + h * hd;
	for (int i = 0; i < hd; i++)
		o[i] = acc[i] / l;
}

__global__ void
k_layernorm(const float *X, const float *w, const float *b, float *Y,
	    int M, int E, float eps)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	if (t >= M)
		return;
	const float *x = X + (size_t)t * E;
	float mean = 0.0f;
	for (int i = 0; i < E; i++)
		mean += x[i];
	mean /= E;
	float var = 0.0f;
	for (int i = 0; i < E; i++) {
		float d = x[i] - mean;
		var += d * d;
	}
	var /= E;
	float inv = 1.0f / sqrtf(var + eps);
	float *y = Y + (size_t)t * E;
	for (int i = 0; i < E; i++)
		y[i] = (x[i] - mean) * inv * w[i] + b[i];
}

// Y[M][N] = X[M][K] @ W[K][N]   (W is [in][out], matching Conv1D)
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

__global__ void
k_addbias(float *Y, const float *b, int M, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= M * N)
		return;
	Y[idx] += b[idx % N];
}

__global__ void
k_add(float *X, const float *Y, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		X[i] += Y[i];
}

// GPT-2 "gelu_new" (tanh approximation).
__global__ void
k_gelu_new(float *X, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;
	float x = X[i];
	float c = 0.7978845608028654f;  /* sqrt(2/pi) */
	X[i] = 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

// Causal multi-head attention over the combined QKV buffer [ntok][3E].
// q = [0,E), k = [E,2E), v = [2E,3E). Online softmax (no length cap).
__global__ void
k_attn_gpt2(const float *QKV, float *O, int ntok, int E, int nh, int hd)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ntok * nh)
		return;
	int h = idx % nh, t = idx / nh;
	float scale = 1.0f / sqrtf((float)hd);
	const float *q = QKV + (size_t)t * 3 * E + h * hd;

	float acc[128];          /* head_dim (64 for gpt2-small) */
	for (int i = 0; i < hd; i++)
		acc[i] = 0.0f;
	float m = -1e30f, l = 0.0f;
	for (int tp = 0; tp <= t; tp++) {
		const float *k = QKV + (size_t)tp * 3 * E + E + h * hd;
		float s = 0.0f;
		for (int i = 0; i < hd; i++)
			s += q[i] * k[i];
		s *= scale;
		float m_new = fmaxf(m, s);
		float corr = expf(m - m_new);
		float p = expf(s - m_new);
		l = l * corr + p;
		const float *v = QKV + (size_t)tp * 3 * E + 2 * E + h * hd;
		for (int i = 0; i < hd; i++)
			acc[i] = acc[i] * corr + p * v[i];
		m = m_new;
	}
	float *o = O + (size_t)t * E + h * hd;
	for (int i = 0; i < hd; i++)
		o[i] = acc[i] / l;
}

// Scatter layer `layer`'s K and V (from the combined QKV buffer) into the fp16
// page-block buffer. GPT-2 has no GQA, so kvdim == E. Layout matches kmodel:
// ((layer*2 + kv)*P + tok_in_page)*E + e.
__global__ void
k_store_kv_gpt2(const float *QKV, __half *blocks, int layer, int L,
		int ntok, int P, int E)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= ntok * E)
		return;
	int e = idx % E, t = idx / E;
	int page = t / P, tip = t % P;
	size_t page_elems = (size_t)2 * L * P * E;
	size_t kbase = (size_t)page * page_elems + ((size_t)(layer * 2 + 0) * P + tip) * E + e;
	size_t vbase = (size_t)page * page_elems + ((size_t)(layer * 2 + 1) * P + tip) * E + e;
	blocks[kbase] = __float2half(QKV[(size_t)t * 3 * E + E + e]);       /* K */
	blocks[vbase] = __float2half(QKV[(size_t)t * 3 * E + 2 * E + e]);   /* V */
}

// Tied lm_head: logits[v] = x_last . wte[v]   (wte is [vocab][E]).
__global__ void
k_lm_head_tied(const float *x_last, const float *wte, float *logits, int E, int V)
{
	int v = blockIdx.x * blockDim.x + threadIdx.x;
	if (v >= V)
		return;
	const float *w = wte + (size_t)v * E;
	float a = 0.0f;
	for (int i = 0; i < E; i++)
		a += x_last[i] * w[i];
	logits[v] = a;
}

// ---- load ------------------------------------------------------------------

static float *
read_dev(FILE *f, size_t n)
{
	std::vector<float> host(n);
	if (fread(host.data(), sizeof(float), n, f) != n) {
		fprintf(stderr, "gpt2_load: short read\n");
		exit(1);
	}
	float *d = nullptr;
	HIP_CHECK(hipMalloc(&d, n * sizeof(float)));
	HIP_CHECK(hipMemcpy(d, host.data(), n * sizeof(float), hipMemcpyHostToDevice));
	return d;
}

Gpt2 *
gpt2_load(const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) {
		fprintf(stderr, "gpt2_load: cannot open %s\n", path);
		return nullptr;
	}
	int32_t hdr[6];
	if (fread(hdr, sizeof(int32_t), 6, f) != 6 || hdr[0] != 0x47505432) {
		fprintf(stderr, "gpt2_load: bad header\n");
		fclose(f);
		return nullptr;
	}
	auto *g = new Gpt2();
	g->L = hdr[1]; g->E = hdr[2]; g->H = hdr[3]; g->P = hdr[4]; g->V = hdr[5];
	int E = g->E;

	g->wte = read_dev(f, (size_t)g->V * E);
	g->wpe = read_dev(f, (size_t)g->P * E);
	g->lnf_w = read_dev(f, E);
	g->lnf_b = read_dev(f, E);
	g->layers.resize(g->L);
	for (auto &Lr : g->layers) {
		Lr.ln1_w   = read_dev(f, E);
		Lr.ln1_b   = read_dev(f, E);
		Lr.cattn_w = read_dev(f, (size_t)E * 3 * E);
		Lr.cattn_b = read_dev(f, (size_t)3 * E);
		Lr.cproj_w = read_dev(f, (size_t)E * E);
		Lr.cproj_b = read_dev(f, E);
		Lr.ln2_w   = read_dev(f, E);
		Lr.ln2_b   = read_dev(f, E);
		Lr.cfc_w   = read_dev(f, (size_t)E * 4 * E);
		Lr.cfc_b   = read_dev(f, (size_t)4 * E);
		Lr.mcproj_w = read_dev(f, (size_t)4 * E * E);
		Lr.mcproj_b = read_dev(f, E);
	}
	fclose(f);
	return g;
}

void
gpt2_free(Gpt2 *g)
{
	if (!g)
		return;
	dfree(g->wte); dfree(g->wpe); dfree(g->lnf_w); dfree(g->lnf_b);
	for (auto &L : g->layers) {
		dfree(L.ln1_w); dfree(L.ln1_b); dfree(L.cattn_w); dfree(L.cattn_b);
		dfree(L.cproj_w); dfree(L.cproj_b); dfree(L.ln2_w); dfree(L.ln2_b);
		dfree(L.cfc_w); dfree(L.cfc_b); dfree(L.mcproj_w); dfree(L.mcproj_b);
	}
	delete g;
}

int gpt2_vocab(const Gpt2 *g) { return g->V; }
int gpt2_layers(const Gpt2 *g) { return g->L; }
int gpt2_embed(const Gpt2 *g) { return g->E; }

size_t
gpt2_kv_block_elems(const Gpt2 *g, int page_tokens)
{
	return (size_t)2 * g->L * page_tokens * g->E;
}

// ---- forward ---------------------------------------------------------------

void
gpt2_forward_logits(Gpt2 *g, const uint32_t *toks, int ntok, float *logits_host,
		    uint16_t *kv_blocks_host, int page_tokens)
{
	int E = g->E, H = g->H, hd = E / H, V = g->V, P = page_tokens;

	uint32_t *d_tok;
	float *x, *xn, *qkv, *attn, *tmp, *h4, *logits;
	HIP_CHECK(hipMalloc(&d_tok, ntok * sizeof(uint32_t)));
	HIP_CHECK(hipMemcpy(d_tok, toks, ntok * sizeof(uint32_t), hipMemcpyHostToDevice));
	HIP_CHECK(hipMalloc(&x,    (size_t)ntok * E * sizeof(float)));
	HIP_CHECK(hipMalloc(&xn,   (size_t)ntok * E * sizeof(float)));
	HIP_CHECK(hipMalloc(&qkv,  (size_t)ntok * 3 * E * sizeof(float)));
	HIP_CHECK(hipMalloc(&attn, (size_t)ntok * E * sizeof(float)));
	HIP_CHECK(hipMalloc(&tmp,  (size_t)ntok * E * sizeof(float)));
	HIP_CHECK(hipMalloc(&h4,   (size_t)ntok * 4 * E * sizeof(float)));
	HIP_CHECK(hipMalloc(&logits, (size_t)V * sizeof(float)));

	__half *d_blocks = nullptr;
	int npages = (ntok + P - 1) / P;
	size_t block_elems = (size_t)2 * g->L * P * E;
	if (kv_blocks_host)
		HIP_CHECK(hipMalloc(&d_blocks, (size_t)npages * block_elems * sizeof(__half)));

	k_embed_pos<<<NBLK(ntok * E), BLK>>>(g->wte, g->wpe, d_tok, x, ntok, E, 0);

	int li = 0;
	for (auto &L : g->layers) {
		k_layernorm<<<NBLK(ntok), BLK>>>(x, L.ln1_w, L.ln1_b, xn, ntok, E, LN_EPS);
		k_mm<<<NBLK(ntok * 3 * E), BLK>>>(xn, L.cattn_w, qkv, ntok, E, 3 * E);
		k_addbias<<<NBLK(ntok * 3 * E), BLK>>>(qkv, L.cattn_b, ntok, 3 * E);
		if (d_blocks)
			k_store_kv_gpt2<<<NBLK(ntok * E), BLK>>>(qkv, d_blocks, li, g->L, ntok, P, E);
		k_attn_gpt2<<<NBLK(ntok * H), BLK>>>(qkv, attn, ntok, E, H, hd);
		k_mm<<<NBLK(ntok * E), BLK>>>(attn, L.cproj_w, tmp, ntok, E, E);
		k_addbias<<<NBLK(ntok * E), BLK>>>(tmp, L.cproj_b, ntok, E);
		k_add<<<NBLK(ntok * E), BLK>>>(x, tmp, ntok * E);

		k_layernorm<<<NBLK(ntok), BLK>>>(x, L.ln2_w, L.ln2_b, xn, ntok, E, LN_EPS);
		k_mm<<<NBLK(ntok * 4 * E), BLK>>>(xn, L.cfc_w, h4, ntok, E, 4 * E);
		k_addbias<<<NBLK(ntok * 4 * E), BLK>>>(h4, L.cfc_b, ntok, 4 * E);
		k_gelu_new<<<NBLK(ntok * 4 * E), BLK>>>(h4, ntok * 4 * E);
		k_mm<<<NBLK(ntok * E), BLK>>>(h4, L.mcproj_w, tmp, ntok, 4 * E, E);
		k_addbias<<<NBLK(ntok * E), BLK>>>(tmp, L.mcproj_b, ntok, E);
		k_add<<<NBLK(ntok * E), BLK>>>(x, tmp, ntok * E);
		li++;
	}

	k_layernorm<<<NBLK(ntok), BLK>>>(x, g->lnf_w, g->lnf_b, xn, ntok, E, LN_EPS);
	k_lm_head_tied<<<NBLK(V), BLK>>>(xn + (size_t)(ntok - 1) * E, g->wte, logits, E, V);
	HIP_CHECK(hipDeviceSynchronize());

	HIP_CHECK(hipMemcpy(logits_host, logits, V * sizeof(float), hipMemcpyDeviceToHost));
	if (kv_blocks_host) {
		HIP_CHECK(hipMemcpy(kv_blocks_host, d_blocks,
				    (size_t)npages * block_elems * sizeof(__half),
				    hipMemcpyDeviceToHost));
		dfree(d_blocks);
	}

	dfree(d_tok); dfree(x); dfree(xn); dfree(qkv); dfree(attn);
	dfree(tmp); dfree(h4); dfree(logits);
}

void
gpt2_generate(Gpt2 *g, std::vector<uint32_t> &toks, int n_new,
	      std::vector<uint32_t> &out)
{
	std::vector<float> logits(g->V);
	for (int s = 0; s < n_new; s++) {
		gpt2_forward_logits(g, toks.data(), (int)toks.size(), logits.data());
		int best = 0;
		for (int v = 1; v < g->V; v++)
			if (logits[v] > logits[best])
				best = v;
		toks.push_back((uint32_t)best);
		out.push_back((uint32_t)best);
	}
}

// ---- incremental KV cache --------------------------------------------------

struct Gpt2KV {
	int maxctx, len;
	std::vector<float *> dK, dV;   /* per layer, [maxctx*E] */
	/* reusable single-token decode scratch */
	uint32_t *d_tok1;
	float *x1, *xn1, *qkv1, *attn1, *tmp1, *h41, *logits;
};

Gpt2KV *
gpt2_kv_create(Gpt2 *g, int maxctx)
{
	int E = g->E;
	auto *kv = new Gpt2KV();
	kv->maxctx = maxctx;
	kv->len = 0;
	kv->dK.resize(g->L);
	kv->dV.resize(g->L);
	for (int l = 0; l < g->L; l++) {
		HIP_CHECK(hipMalloc(&kv->dK[l], (size_t)maxctx * E * sizeof(float)));
		HIP_CHECK(hipMalloc(&kv->dV[l], (size_t)maxctx * E * sizeof(float)));
	}
	HIP_CHECK(hipMalloc(&kv->d_tok1, sizeof(uint32_t)));
	HIP_CHECK(hipMalloc(&kv->x1,   (size_t)E * sizeof(float)));
	HIP_CHECK(hipMalloc(&kv->xn1,  (size_t)E * sizeof(float)));
	HIP_CHECK(hipMalloc(&kv->qkv1, (size_t)3 * E * sizeof(float)));
	HIP_CHECK(hipMalloc(&kv->attn1,(size_t)E * sizeof(float)));
	HIP_CHECK(hipMalloc(&kv->tmp1, (size_t)E * sizeof(float)));
	HIP_CHECK(hipMalloc(&kv->h41,  (size_t)4 * E * sizeof(float)));
	HIP_CHECK(hipMalloc(&kv->logits, (size_t)g->V * sizeof(float)));
	return kv;
}

void
gpt2_kv_free(Gpt2KV *kv)
{
	if (!kv)
		return;
	for (auto p : kv->dK) dfree(p);
	for (auto p : kv->dV) dfree(p);
	dfree(kv->d_tok1); dfree(kv->x1); dfree(kv->xn1); dfree(kv->qkv1);
	dfree(kv->attn1); dfree(kv->tmp1); dfree(kv->h41); dfree(kv->logits);
	delete kv;
}

// Prefill: run the batched forward over the prompt, appending its K/V to the
// cache. Returns last-token logits. Sets kv->len = ntok.
static void
gpt2_prefill(Gpt2 *g, Gpt2KV *kv, const uint32_t *toks, int ntok, float *logits_host)
{
	int E = g->E, H = g->H, hd = E / H, V = g->V;
	uint32_t *d_tok;
	float *x, *xn, *qkv, *attn, *tmp, *h4, *logits;
	HIP_CHECK(hipMalloc(&d_tok, ntok * sizeof(uint32_t)));
	HIP_CHECK(hipMemcpy(d_tok, toks, ntok * sizeof(uint32_t), hipMemcpyHostToDevice));
	HIP_CHECK(hipMalloc(&x,    (size_t)ntok * E * sizeof(float)));
	HIP_CHECK(hipMalloc(&xn,   (size_t)ntok * E * sizeof(float)));
	HIP_CHECK(hipMalloc(&qkv,  (size_t)ntok * 3 * E * sizeof(float)));
	HIP_CHECK(hipMalloc(&attn, (size_t)ntok * E * sizeof(float)));
	HIP_CHECK(hipMalloc(&tmp,  (size_t)ntok * E * sizeof(float)));
	HIP_CHECK(hipMalloc(&h4,   (size_t)ntok * 4 * E * sizeof(float)));
	HIP_CHECK(hipMalloc(&logits, (size_t)V * sizeof(float)));

	k_embed_pos<<<NBLK(ntok * E), BLK>>>(g->wte, g->wpe, d_tok, x, ntok, E, 0);
	int li = 0;
	for (auto &L : g->layers) {
		k_layernorm<<<NBLK(ntok), BLK>>>(x, L.ln1_w, L.ln1_b, xn, ntok, E, LN_EPS);
		k_mm<<<NBLK(ntok * 3 * E), BLK>>>(xn, L.cattn_w, qkv, ntok, E, 3 * E);
		k_addbias<<<NBLK(ntok * 3 * E), BLK>>>(qkv, L.cattn_b, ntok, 3 * E);
		k_append_kv<<<NBLK(ntok * E), BLK>>>(qkv, kv->dK[li], kv->dV[li], 0, ntok, E);
		k_attn_gpt2<<<NBLK(ntok * H), BLK>>>(qkv, attn, ntok, E, H, hd);
		k_mm<<<NBLK(ntok * E), BLK>>>(attn, L.cproj_w, tmp, ntok, E, E);
		k_addbias<<<NBLK(ntok * E), BLK>>>(tmp, L.cproj_b, ntok, E);
		k_add<<<NBLK(ntok * E), BLK>>>(x, tmp, ntok * E);
		k_layernorm<<<NBLK(ntok), BLK>>>(x, L.ln2_w, L.ln2_b, xn, ntok, E, LN_EPS);
		k_mm<<<NBLK(ntok * 4 * E), BLK>>>(xn, L.cfc_w, h4, ntok, E, 4 * E);
		k_addbias<<<NBLK(ntok * 4 * E), BLK>>>(h4, L.cfc_b, ntok, 4 * E);
		k_gelu_new<<<NBLK(ntok * 4 * E), BLK>>>(h4, ntok * 4 * E);
		k_mm<<<NBLK(ntok * E), BLK>>>(h4, L.mcproj_w, tmp, ntok, 4 * E, E);
		k_addbias<<<NBLK(ntok * E), BLK>>>(tmp, L.mcproj_b, ntok, E);
		k_add<<<NBLK(ntok * E), BLK>>>(x, tmp, ntok * E);
		li++;
	}
	k_layernorm<<<NBLK(ntok), BLK>>>(x, g->lnf_w, g->lnf_b, xn, ntok, E, LN_EPS);
	k_lm_head_tied<<<NBLK(V), BLK>>>(xn + (size_t)(ntok - 1) * E, g->wte, logits, E, V);
	HIP_CHECK(hipDeviceSynchronize());
	HIP_CHECK(hipMemcpy(logits_host, logits, V * sizeof(float), hipMemcpyDeviceToHost));
	kv->len = ntok;
	dfree(d_tok); dfree(x); dfree(xn); dfree(qkv); dfree(attn);
	dfree(tmp); dfree(h4); dfree(logits);
}

// Decode one token at position kv->len using the cache; returns its logits.
static void
gpt2_decode_step(Gpt2 *g, Gpt2KV *kv, uint32_t token, float *logits_host)
{
	int E = g->E, H = g->H, hd = E / H, V = g->V;
	int pos = kv->len;
	HIP_CHECK(hipMemcpy(kv->d_tok1, &token, sizeof(uint32_t), hipMemcpyHostToDevice));
	k_embed_pos<<<NBLK(E), BLK>>>(g->wte, g->wpe, kv->d_tok1, kv->x1, 1, E, pos);
	int li = 0;
	for (auto &L : g->layers) {
		k_layernorm<<<1, BLK>>>(kv->x1, L.ln1_w, L.ln1_b, kv->xn1, 1, E, LN_EPS);
		k_mm<<<NBLK(3 * E), BLK>>>(kv->xn1, L.cattn_w, kv->qkv1, 1, E, 3 * E);
		k_addbias<<<NBLK(3 * E), BLK>>>(kv->qkv1, L.cattn_b, 1, 3 * E);
		k_append_kv<<<NBLK(E), BLK>>>(kv->qkv1, kv->dK[li], kv->dV[li], pos, 1, E);
		k_attn_decode<<<NBLK(H), BLK>>>(kv->qkv1, kv->dK[li], kv->dV[li],
						kv->attn1, pos + 1, H, hd, E);
		k_mm<<<NBLK(E), BLK>>>(kv->attn1, L.cproj_w, kv->tmp1, 1, E, E);
		k_addbias<<<NBLK(E), BLK>>>(kv->tmp1, L.cproj_b, 1, E);
		k_add<<<NBLK(E), BLK>>>(kv->x1, kv->tmp1, E);
		k_layernorm<<<1, BLK>>>(kv->x1, L.ln2_w, L.ln2_b, kv->xn1, 1, E, LN_EPS);
		k_mm<<<NBLK(4 * E), BLK>>>(kv->xn1, L.cfc_w, kv->h41, 1, E, 4 * E);
		k_addbias<<<NBLK(4 * E), BLK>>>(kv->h41, L.cfc_b, 1, 4 * E);
		k_gelu_new<<<NBLK(4 * E), BLK>>>(kv->h41, 4 * E);
		k_mm<<<NBLK(E), BLK>>>(kv->h41, L.mcproj_w, kv->tmp1, 1, 4 * E, E);
		k_addbias<<<NBLK(E), BLK>>>(kv->tmp1, L.mcproj_b, 1, E);
		k_add<<<NBLK(E), BLK>>>(kv->x1, kv->tmp1, E);
		li++;
	}
	k_layernorm<<<1, BLK>>>(kv->x1, g->lnf_w, g->lnf_b, kv->xn1, 1, E, LN_EPS);
	k_lm_head_tied<<<NBLK(V), BLK>>>(kv->xn1, g->wte, kv->logits, E, V);
	HIP_CHECK(hipDeviceSynchronize());
	HIP_CHECK(hipMemcpy(logits_host, kv->logits, V * sizeof(float), hipMemcpyDeviceToHost));
	kv->len = pos + 1;
}

// Pick a token: argmax if temp<=0, else sample from softmax(logits/temp).
static uint32_t
pick_token(const std::vector<float> &logits, int V, float temp, uint64_t &rng)
{
	if (temp <= 0.0f) {
		int best = 0;
		for (int v = 1; v < V; v++)
			if (logits[v] > logits[best])
				best = v;
		return (uint32_t)best;
	}
	float mx = logits[0];
	for (int v = 1; v < V; v++)
		mx = fmaxf(mx, logits[v]);
	double sum = 0.0;
	std::vector<double> p(V);
	for (int v = 0; v < V; v++) {
		p[v] = exp((logits[v] - mx) / temp);
		sum += p[v];
	}
	rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
	double u = ((double)(rng >> 11) / (double)(1ULL << 53)) * sum;
	double acc = 0.0;
	for (int v = 0; v < V; v++) {
		acc += p[v];
		if (u < acc)
			return (uint32_t)v;
	}
	return (uint32_t)(V - 1);
}

static void
decode_loop(Gpt2 *g, Gpt2KV *kv, std::vector<uint32_t> &toks, int n_new,
	    std::vector<uint32_t> &out, std::vector<float> &logits,
	    float temp, uint64_t seed)
{
	uint64_t rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
	for (int s = 0; s < n_new; s++) {
		uint32_t next = pick_token(logits, g->V, temp, rng);
		toks.push_back(next);
		out.push_back(next);
		gpt2_decode_step(g, kv, next, logits.data());
	}
}

void
gpt2_generate_cached(Gpt2 *g, Gpt2KV *kv, std::vector<uint32_t> &toks,
		     int n_new, std::vector<uint32_t> &out, float temp, uint64_t seed)
{
	std::vector<float> logits(g->V);
	gpt2_prefill(g, kv, toks.data(), (int)toks.size(), logits.data());
	decode_loop(g, kv, toks, n_new, out, logits, temp, seed);
}

// ---- warm-start: load content-addressed prefix KV, skip its prefill --------

// Load n_load cached positions' K/V (fp16 page blocks) into a layer's fp32 cache.
__global__ void
k_load_cache(const __half *blocks, float *dK, float *dV, int layer, int L,
	     int n_load, int P, int E)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n_load * E)
		return;
	int e = idx % E, pos = idx / E;
	int page = pos / P, tip = pos % P;
	size_t page_elems = (size_t)2 * L * P * E;
	size_t kb = (size_t)page * page_elems + ((size_t)(layer * 2 + 0) * P + tip) * E + e;
	size_t vb = (size_t)page * page_elems + ((size_t)(layer * 2 + 1) * P + tip) * E + e;
	dK[(size_t)pos * E + e] = __half2float(blocks[kb]);
	dV[(size_t)pos * E + e] = __half2float(blocks[vb]);
}

// Pack a layer's fp32 cache [0,len) into the fp16 page-block buffer (the inverse
// of k_load_cache) so generated KV can be stored content-addressed without an
// extra forward pass.
__global__ void
k_pack_blocks(const float *dK, const float *dV, __half *blocks, int layer, int L,
	      int len, int P, int E)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= len * E)
		return;
	int e = idx % E, pos = idx / E;
	int page = pos / P, tip = pos % P;
	size_t page_elems = (size_t)2 * L * P * E;
	size_t kb = (size_t)page * page_elems + ((size_t)(layer * 2 + 0) * P + tip) * E + e;
	size_t vb = (size_t)page * page_elems + ((size_t)(layer * 2 + 1) * P + tip) * E + e;
	blocks[kb] = __float2half(dK[(size_t)pos * E + e]);
	blocks[vb] = __float2half(dV[(size_t)pos * E + e]);
}

// Pack the whole KV cache [0,len) into fp16 page blocks (host buffer sized
// ceil(len/P) * gpt2_kv_block_elems). Unused tail slots of the last page are 0.
void
gpt2_kv_to_blocks(Gpt2 *g, Gpt2KV *kv, int page_tokens, uint16_t *blocks_host)
{
	int E = g->E, len = kv->len, P = page_tokens;
	int npages = (len + P - 1) / P;
	size_t block_elems = (size_t)2 * g->L * P * E;
	__half *d_blocks;
	HIP_CHECK(hipMalloc(&d_blocks, (size_t)npages * block_elems * sizeof(__half)));
	HIP_CHECK(hipMemset(d_blocks, 0, (size_t)npages * block_elems * sizeof(__half)));
	for (int l = 0; l < g->L; l++)
		k_pack_blocks<<<NBLK(len * E), BLK>>>(kv->dK[l], kv->dV[l], d_blocks,
						      l, g->L, len, P, E);
	HIP_CHECK(hipDeviceSynchronize());
	HIP_CHECK(hipMemcpy(blocks_host, d_blocks,
			    (size_t)npages * block_elems * sizeof(__half),
			    hipMemcpyDeviceToHost));
	dfree(d_blocks);
}

void
gpt2_generate_warm(Gpt2 *g, Gpt2KV *kv, const uint16_t *prefix_blocks,
		   int n_load, int page_tokens, std::vector<uint32_t> &toks,
		   int n_new, std::vector<uint32_t> &out, float temp, uint64_t seed)
{
	int E = g->E, ntok = (int)toks.size();
	int npages = (n_load + page_tokens - 1) / page_tokens;  /* only the cached pages */
	size_t block_elems = (size_t)2 * g->L * page_tokens * E;

	/* stage the cached prefix pages to device and load into the KV cache */
	__half *d_blocks;
	HIP_CHECK(hipMalloc(&d_blocks, (size_t)npages * block_elems * sizeof(__half)));
	HIP_CHECK(hipMemcpy(d_blocks, prefix_blocks,
			    (size_t)npages * block_elems * sizeof(__half),
			    hipMemcpyHostToDevice));
	for (int l = 0; l < g->L; l++)
		k_load_cache<<<NBLK(n_load * E), BLK>>>(d_blocks, kv->dK[l], kv->dV[l],
							l, g->L, n_load, page_tokens, E);
	HIP_CHECK(hipDeviceSynchronize());
	dfree(d_blocks);
	kv->len = n_load;   /* prefix loaded; its prefill is SKIPPED */

	/* process the remaining prompt tokens (incl. the last) to get logits */
	std::vector<float> logits(g->V);
	for (int t = n_load; t < ntok; t++)
		gpt2_decode_step(g, kv, toks[t], logits.data());

	decode_loop(g, kv, toks, n_new, out, logits, temp, seed);
}
