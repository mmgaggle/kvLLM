/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * kllm-serve — /dev/kllm backed by real IBM Granite 4.0.
 *
 * The device speaks TOKEN IDS (the BPE tokenizer lives in the userspace client,
 * per the locked decision). Request on a single open fd:
 *
 *     "<n_new> <id> <id> ..."   (first int = tokens to generate, rest = prompt)
 *
 * Response:
 *     "<id> <id> ...\n"         (generated token ids)
 *     "# pages hits H misses M store S\n"   (content-addressed KV stats)
 *
 * Generation runs on the GPU; each request's prefix pages are content-addressed
 * (prefix Merkle hash) into a process-global KV store, so two requests sharing a
 * prefix share KV blocks — the thesis, on a real model, through the char device.
 *
 * Built host-only (g++) so the cuse op table (host fn pointers) doesn't hit the
 * hipcc device pass. Run: sudo ./gpu/kllm-serve -f
 */
#define FUSE_USE_VERSION 31

#include <cuse_lowlevel.h>
#include <fuse_opt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include "granite.h"
#include "chash.h"
#include "kvstore.h"

#define PAGE_TOKENS 8   /* smaller pages => prompts have full leading pages to reuse */

static Granite          *g_model;
static kllm_kvstore  *g_ks;
static kllm_key       g_seed;

struct fh_state {
	std::string  req;
	std::string  resp;
	size_t       rpos = 0;
	bool         done = false;
};

/* Page key for tokens[p0, p0+pn) given the running Merkle parent. */
static kllm_key
page_key(kllm_key parent, const std::vector<uint32_t> &toks, int p0, int pn)
{
	return kllm_chash_chain(parent, &toks[p0], (size_t)pn * sizeof(uint32_t));
}

static void
serve_request(fh_state *st)
{
	std::istringstream in(st->req);
	long n_new = 0;
	float temp = 0.0f;
	in >> n_new >> temp;            /* "<n_new> <temp> <id>..." (temp 0 = greedy) */
	std::vector<uint32_t> toks;
	long id;
	while (in >> id)
		toks.push_back((uint32_t)id);

	std::ostringstream out;
	if (toks.empty() || n_new <= 0) {
		out << "\n# error: expected '<n_new> <temp> <id>...'\n";
		st->resp = out.str();
		st->done = true;
		return;
	}

	int ntok = (int)toks.size();
	size_t be = granite_kv_block_elems(g_model, PAGE_TOKENS);

	/* How many LEADING prompt pages are already in the content store? Those KV
	 * blocks can be loaded and their prefill skipped. */
	int nprompt_pages = (ntok + PAGE_TOKENS - 1) / PAGE_TOKENS;
	std::vector<kllm_key> pkeys;
	kllm_key parent = g_seed;
	int hit_pages = 0;
	bool counting = true;
	for (int p = 0; p < nprompt_pages; p++) {
		int p0 = p * PAGE_TOKENS;
		int pn = (ntok - p0) < PAGE_TOKENS ? (ntok - p0) : PAGE_TOKENS;
		kllm_key k = page_key(parent, toks, p0, pn);
		pkeys.push_back(k);
		parent = k;
		if (counting && kllm_kvstore_exist(g_ks, k))
			hit_pages++;
		else
			counting = false;
	}
	int n_load = hit_pages * PAGE_TOKENS;
	if (n_load > ntok - 1)
		n_load = ntok - 1;      /* always recompute the last token (need logits) */
	if (n_load < 0)
		n_load = 0;

	/* Generate: warm-start from the cached prefix if we have one. */
	GraniteKV *kv = granite_kv_create(g_model, ntok + (int)n_new + 8);
	std::vector<uint32_t> gen;
	if (n_load > 0) {
		int npl = (n_load + PAGE_TOKENS - 1) / PAGE_TOKENS;
		std::vector<uint16_t> load_blocks((size_t)npl * be);
		for (int p = 0; p < npl; p++) {
			uint32_t vlen = 0;
			kllm_kvstore_retrieve(g_ks, pkeys[p], &load_blocks[(size_t)p * be],
					      (uint32_t)(be * sizeof(uint16_t)), &vlen);
		}
		granite_generate_warm(g_model, kv, load_blocks.data(), n_load, PAGE_TOKENS,
				   toks, (int)n_new, gen, temp, 1234ULL);
	} else {
		granite_generate_cached(g_model, kv, toks, (int)n_new, gen, temp, 1234ULL);
	}

	/* Store the full sequence's pages from the cache (no extra forward). */
	int full = (int)toks.size();
	int nfull = (full + PAGE_TOKENS - 1) / PAGE_TOKENS;
	std::vector<uint16_t> full_blocks((size_t)nfull * be);
	granite_kv_to_blocks(g_model, kv, PAGE_TOKENS, full_blocks.data());
	granite_kv_free(kv);

	parent = g_seed;
	int stored = 0, reused = 0;
	for (int p = 0; p < nfull; p++) {
		int p0 = p * PAGE_TOKENS;
		int pn = (full - p0) < PAGE_TOKENS ? (full - p0) : PAGE_TOKENS;
		kllm_key k = page_key(parent, toks, p0, pn);
		parent = k;
		if (kllm_kvstore_exist(g_ks, k))
			reused++;
		else {
			kllm_kvstore_store(g_ks, k, &full_blocks[(size_t)p * be],
					   (uint32_t)(be * sizeof(uint16_t)));
			stored++;
		}
	}

	for (size_t i = 0; i < gen.size(); i++)
		out << gen[i] << (i + 1 < gen.size() ? " " : "");
	out << "\n# prefill_skipped " << n_load << " of " << ntok
	    << " | pages +" << stored << " reused " << reused
	    << " | store_total " << kllm_kvstore_count(g_ks) << "\n";
	st->resp = out.str();
	st->done = true;
}

static void
kllm_open(fuse_req_t req, struct fuse_file_info *fi)
{
	fi->fh = (uintptr_t)(new fh_state());
	fi->nonseekable = 1;
	fi->direct_io = 1;
	fuse_reply_open(req, fi);
}

static void
kllm_release(fuse_req_t req, struct fuse_file_info *fi)
{
	delete (fh_state *)(uintptr_t)fi->fh;
	fuse_reply_err(req, 0);
}

static void
kllm_write(fuse_req_t req, const char *buf, size_t size, off_t off,
	   struct fuse_file_info *fi)
{
	(void)off;
	auto *st = (fh_state *)(uintptr_t)fi->fh;
	if (st->done) {              /* a new write starts a fresh request */
		st->req.clear();
		st->resp.clear();
		st->rpos = 0;
		st->done = false;
	}
	st->req.append(buf, size);
	fuse_reply_write(req, size);
}

static void
kllm_read(fuse_req_t req, size_t size, off_t off, struct fuse_file_info *fi)
{
	(void)off;
	auto *st = (fh_state *)(uintptr_t)fi->fh;
	if (!st->done)
		serve_request(st);
	if (st->rpos >= st->resp.size()) {
		fuse_reply_buf(req, NULL, 0);
		return;
	}
	size_t avail = st->resp.size() - st->rpos;
	size_t n = size < avail ? size : avail;
	fuse_reply_buf(req, st->resp.data() + st->rpos, n);
	st->rpos += n;
}

int
main(int argc, char **argv)
{
	struct cuse_lowlevel_ops kllm_clop;
	memset(&kllm_clop, 0, sizeof(kllm_clop));
	kllm_clop.open = kllm_open;
	kllm_clop.read = kllm_read;
	kllm_clop.write = kllm_write;
	kllm_clop.release = kllm_release;

	const char *wts = getenv("KLLM_WEIGHTS");
	g_model = granite_load(wts ? wts : "models/granite/granite.weights");
	if (!g_model) {
		fprintf(stderr, "kllm-serve: failed to load GPT-2 weights\n");
		return 1;
	}
	const char *ctx = "granite-4.0-350m|fp32";
	g_seed = kllm_chash(ctx, strlen(ctx));
	g_ks = kllm_kvstore_create();

	const char *dev_info_argv[] = { "DEVNAME=kllm" };
	struct cuse_info ci;
	memset(&ci, 0, sizeof(ci));
	ci.dev_info_argc = 1;
	ci.dev_info_argv = dev_info_argv;
	ci.flags = CUSE_UNRESTRICTED_IOCTL;

	fprintf(stderr, "kllm-serve: Granite ready, serving /dev/kllm\n");
	return cuse_lowlevel_main(argc, argv, &ci, &kllm_clop, NULL);
}
