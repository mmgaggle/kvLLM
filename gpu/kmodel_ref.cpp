/* SPDX-License-Identifier: BSD-3-Clause */
/* CPU reference forward (fp32) — the parity oracle for the HIP kernels. */
#include <cmath>
#include <vector>
#include "kmodel.h"

KCfg
ktiny_cfg(void)
{
	KCfg c;
	c.hidden = 64;
	c.n_layers = 2;
	c.n_heads = 4;
	c.n_kv_heads = 2;
	c.head_dim = 16;       /* hidden == n_heads*head_dim == 64 */
	c.intermediate = 128;
	c.vocab = 256;        /* covers the byte-level tokenizer */
	c.page_tokens = 4;
	c.rope_theta = 10000.0f;
	c.rms_eps = 1e-5f;
	return c;
}

static float
rnd(uint64_t &s)
{
	s = s * 6364136223846793005ULL + 1442695040888963407ULL;
	uint32_t x = (uint32_t)(s >> 32);
	return ((float)x / (float)UINT32_MAX) * 2.0f - 1.0f;  /* [-1,1] */
}

static void
fill(std::vector<float> &v, size_t n, uint64_t &s, float scale)
{
	v.resize(n);
	for (size_t i = 0; i < n; i++)
		v[i] = rnd(s) * scale;
}

static void
fill_norm(std::vector<float> &v, size_t n, uint64_t &s)
{
	v.resize(n);
	for (size_t i = 0; i < n; i++)
		v[i] = 1.0f + rnd(s) * 0.02f;  /* RMSNorm weights ~1 */
}

void
host_model_init_random(HostModel &m, const KCfg &cfg, uint64_t seed)
{
	m.cfg = cfg;
	uint64_t s = seed;
	int H = cfg.hidden, KV = cfg.n_kv_heads * cfg.head_dim, I = cfg.intermediate;

	fill(m.tok_emb, (size_t)cfg.vocab * H, s, 0.05f);
	fill_norm(m.final_norm, H, s);
	fill(m.lm_head, (size_t)H * cfg.vocab, s, 0.05f);

	m.layers.resize(cfg.n_layers);
	for (auto &L : m.layers) {
		fill_norm(L.attn_norm, H, s);
		fill(L.wq, (size_t)H * H, s, 0.05f);
		fill(L.wk, (size_t)H * KV, s, 0.05f);
		fill(L.wv, (size_t)H * KV, s, 0.05f);
		fill(L.wo, (size_t)H * H, s, 0.05f);
		fill_norm(L.ffn_norm, H, s);
		fill(L.wg, (size_t)H * I, s, 0.05f);
		fill(L.wu, (size_t)H * I, s, 0.05f);
		fill(L.wd, (size_t)I * H, s, 0.05f);
	}
}

/* y[M][N] = x[M][K] @ W[K][N] */
static void
mm(const float *X, const float *W, float *Y, int M, int K, int N)
{
	for (int m = 0; m < M; m++)
		for (int n = 0; n < N; n++) {
			float a = 0.0f;
			for (int k = 0; k < K; k++)
				a += X[m * K + k] * W[k * N + n];
			Y[m * N + n] = a;
		}
}

static void
rms(const float *X, const float *w, float *Y, int M, int H, float eps)
{
	for (int t = 0; t < M; t++) {
		float ss = 0.0f;
		for (int i = 0; i < H; i++) {
			float v = X[t * H + i];
			ss += v * v;
		}
		float sc = 1.0f / sqrtf(ss / H + eps);
		for (int i = 0; i < H; i++)
			Y[t * H + i] = X[t * H + i] * sc * w[i];
	}
}

/* Interleaved-pair RoPE at absolute position = token index. */
static void
rope(float *B, int M, int nh, int hd, float theta)
{
	int half = hd / 2;
	for (int t = 0; t < M; t++)
		for (int h = 0; h < nh; h++) {
			float *base = B + (size_t)(t * nh + h) * hd;
			for (int p = 0; p < half; p++) {
				float freq = powf(theta, -2.0f * p / hd);
				float ang = t * freq;
				float c = cosf(ang), s = sinf(ang);
				float x0 = base[2 * p], x1 = base[2 * p + 1];
				base[2 * p]     = x0 * c - x1 * s;
				base[2 * p + 1] = x0 * s + x1 * c;
			}
		}
}

/* scatter layer l's roped K and plain V into the page-block buffer */
static void
store_kv(std::vector<float> &blocks, const KCfg &c, int l,
	 const float *k, const float *v, int ntok)
{
	int hd = c.head_dim, kvdim = c.n_kv_heads * hd, P = c.page_tokens, L = c.n_layers;
	size_t page_elems = kv_block_elems(c);
	for (int t = 0; t < ntok; t++) {
		int page = t / P, tip = t % P;
		size_t kbase = page * page_elems + ((size_t)(l * 2 + 0) * P + tip) * kvdim;
		size_t vbase = page * page_elems + ((size_t)(l * 2 + 1) * P + tip) * kvdim;
		for (int e = 0; e < kvdim; e++) {
			blocks[kbase + e] = k[(size_t)t * kvdim + e];
			blocks[vbase + e] = v[(size_t)t * kvdim + e];
		}
	}
}

void
cpu_forward_logits(const HostModel &m, const uint32_t *tokens, int ntok,
		   std::vector<float> &logits_last, std::vector<float> *kv_blocks)
{
	const KCfg &c = m.cfg;
	int H = c.hidden, hd = c.head_dim, nh = c.n_heads, nkv = c.n_kv_heads;
	int KV = nkv * hd, I = c.intermediate, group = nh / nkv;
	float scale = 1.0f / sqrtf((float)hd);

	if (kv_blocks)
		kv_blocks->assign((size_t)kv_num_pages(ntok, c.page_tokens) *
				  kv_block_elems(c), 0.0f);
	int layer_idx = 0;

	std::vector<float> x((size_t)ntok * H), xn((size_t)ntok * H);
	std::vector<float> q((size_t)ntok * H), k((size_t)ntok * KV), v((size_t)ntok * KV);
	std::vector<float> attn((size_t)ntok * H), tmp((size_t)ntok * H);
	std::vector<float> gate((size_t)ntok * I), up((size_t)ntok * I), act((size_t)ntok * I);

	/* embed */
	for (int t = 0; t < ntok; t++)
		for (int i = 0; i < H; i++)
			x[t * H + i] = m.tok_emb[(size_t)tokens[t] * H + i];

	for (const auto &L : m.layers) {
		rms(x.data(), L.attn_norm.data(), xn.data(), ntok, H, c.rms_eps);
		mm(xn.data(), L.wq.data(), q.data(), ntok, H, H);
		mm(xn.data(), L.wk.data(), k.data(), ntok, H, KV);
		mm(xn.data(), L.wv.data(), v.data(), ntok, H, KV);
		rope(q.data(), ntok, nh, hd, c.rope_theta);
		rope(k.data(), ntok, nkv, hd, c.rope_theta);

		/* cache the roped K and plain V for this layer in block layout */
		if (kv_blocks)
			store_kv(*kv_blocks, c, layer_idx, k.data(), v.data(), ntok);

		/* causal GQA attention */
		for (int t = 0; t < ntok; t++)
			for (int h = 0; h < nh; h++) {
				int kvh = h / group;
				const float *qp = &q[(size_t)(t * nh + h) * hd];
				std::vector<float> sc(t + 1);
				float maxv = -1e30f;
				for (int tp = 0; tp <= t; tp++) {
					const float *kp = &k[(size_t)(tp * nkv + kvh) * hd];
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
				float *op = &attn[(size_t)(t * nh + h) * hd];
				for (int i = 0; i < hd; i++)
					op[i] = 0.0f;
				for (int tp = 0; tp <= t; tp++) {
					const float *vp = &v[(size_t)(tp * nkv + kvh) * hd];
					float w = sc[tp] / sum;
					for (int i = 0; i < hd; i++)
						op[i] += w * vp[i];
				}
			}

		mm(attn.data(), L.wo.data(), tmp.data(), ntok, H, H);
		for (size_t i = 0; i < (size_t)ntok * H; i++)
			x[i] += tmp[i];

		rms(x.data(), L.ffn_norm.data(), xn.data(), ntok, H, c.rms_eps);
		mm(xn.data(), L.wg.data(), gate.data(), ntok, H, I);
		mm(xn.data(), L.wu.data(), up.data(), ntok, H, I);
		for (size_t i = 0; i < (size_t)ntok * I; i++) {
			float g = gate[i];
			act[i] = (g / (1.0f + expf(-g))) * up[i];
		}
		mm(act.data(), L.wd.data(), tmp.data(), ntok, I, H);
		for (size_t i = 0; i < (size_t)ntok * H; i++)
			x[i] += tmp[i];

		layer_idx++;
	}

	rms(x.data(), m.final_norm.data(), xn.data(), ntok, H, c.rms_eps);

	/* logits for the last position only */
	logits_last.assign(c.vocab, 0.0f);
	int last = ntok - 1;
	for (int n = 0; n < c.vocab; n++) {
		float a = 0.0f;
		for (int kk = 0; kk < H; kk++)
			a += xn[(size_t)last * H + kk] * m.lm_head[(size_t)kk * c.vocab + n];
		logits_last[n] = a;
	}
}
