/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Model + forward contract.
 *
 * A small Llama-style decoder used to validate the HIP forward pass against a
 * CPU reference of identical math. fp32 throughout for clean parity; fp16 +
 * KV-block-layout writeback is a later addition. Weights are deterministic-random here;
 * real safetensors loading is a follow-on once the kernels are proven.
 *
 * Convention: weight matrices are row-major [in][out], so y = x @ W means
 * y[n] = sum_k x[k] * W[k*OUT + n].
 */
#ifndef KLLM_KMODEL_H
#define KLLM_KMODEL_H

#include <cstdint>
#include <vector>

struct KCfg {
	int   hidden;        /* = n_heads * head_dim */
	int   n_layers;
	int   n_heads;       /* query heads */
	int   n_kv_heads;    /* GQA: n_heads % n_kv_heads == 0 */
	int   head_dim;
	int   intermediate;  /* MLP inner dim */
	int   vocab;
	int   page_tokens;   /* tokens per KV page (paged attention block) */
	float rope_theta;    /* e.g. 10000 */
	float rms_eps;       /* e.g. 1e-5 */
};

/*
 * KV page-block layout (matches the engine's block_bytes).
 * One page holds page_tokens consecutive positions across ALL layers, so a
 * single fetch faults in everything attention needs for that token range:
 *
 *   block element index = ((layer*2 + kv)*page_tokens + tok_in_page)*kvdim
 *                         + head*head_dim + d
 *   where kv = 0 for roped-K, 1 for V; kvdim = n_kv_heads*head_dim.
 */
static inline int
kv_num_pages(int ntok, int page_tokens)
{
	return (ntok + page_tokens - 1) / page_tokens;
}

static inline size_t
kv_block_elems(const KCfg &c)  /* fp16 elements in one page block */
{
	return (size_t)2 * c.n_layers * c.page_tokens * c.n_kv_heads * c.head_dim;
}

struct HostLayer {
	std::vector<float> attn_norm;  /* [hidden] */
	std::vector<float> wq;         /* [hidden][n_heads*head_dim] */
	std::vector<float> wk;         /* [hidden][n_kv_heads*head_dim] */
	std::vector<float> wv;         /* [hidden][n_kv_heads*head_dim] */
	std::vector<float> wo;         /* [n_heads*head_dim][hidden] */
	std::vector<float> ffn_norm;   /* [hidden] */
	std::vector<float> wg;         /* [hidden][intermediate]  (gate) */
	std::vector<float> wu;         /* [hidden][intermediate]  (up)   */
	std::vector<float> wd;         /* [intermediate][hidden]  (down) */
};

struct HostModel {
	KCfg cfg;
	std::vector<float> tok_emb;    /* [vocab][hidden] */
	std::vector<float> final_norm; /* [hidden] */
	std::vector<float> lm_head;    /* [hidden][vocab] */
	std::vector<HostLayer> layers;
};

/* A small reference config for parity testing. */
KCfg ktiny_cfg(void);

/* Deterministically fill a model with small random weights. */
void host_model_init_random(HostModel &m, const KCfg &cfg, uint64_t seed);

/* CPU reference forward. Fills logits_last with the vocab-sized logits of the
 * final token position. If kv_blocks != null, also fills it (fp32) with the
 * KV page blocks in the layout above: kv_num_pages * kv_block_elems entries. */
void cpu_forward_logits(const HostModel &m, const uint32_t *tokens, int ntok,
			std::vector<float> &logits_last,
			std::vector<float> *kv_blocks = nullptr);

/* ---- GPU side (defined in kmodel_hip.hip) ---- */

struct HipModel;  /* opaque: weights in VRAM */

HipModel *hip_upload(const HostModel &m);
void hip_free(HipModel *hm);

/* Run the HIP forward; writes vocab floats for the last position into
 * logits_last_host (caller-allocated, cfg.vocab floats). If kv_blocks_host !=
 * null, also writes the fp16 KV page blocks (raw uint16 bits) there:
 * kv_num_pages * kv_block_elems entries. */
void hip_forward_logits(HipModel *hm, const uint32_t *tokens, int ntok,
			float *logits_last_host, uint16_t *kv_blocks_host = nullptr);

/* Continuation forward: compute only tokens [prefix_len, ntok) using the cached
 * fp16 prefix KV blocks (prefix_len/page_tokens pages) as attention context.
 * Writes vocab logits for the last position. prefix_len must be a multiple of
 * page_tokens. This is the compute-saving prefix-cache path. */
void hip_forward_cont(HipModel *hm, const uint32_t *tokens, int ntok,
		      int prefix_len, const uint16_t *prefix_blocks_host,
		      float *logits_last_host);

#endif /* KLLM_KMODEL_H */
