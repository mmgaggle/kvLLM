/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * /dev/kllm — the character-device head, via CUSE.
 *
 * Protocol: write() a prompt, read() a cache report. Each fd holds an
 * independent prompt buffer; the report is computed on first read and describes
 * how the request hit/missed against the process-global, content-addressed KV
 * store. Writing again after a read starts a fresh request.
 *
 * The KV store is shared across all opens — that is the point: two clients
 * sending the same prefix share blocks. Real completions (sampled tokens)
 * replace the report when wired to the GPU engine.
 *
 * Requires the cuse kernel module (modprobe cuse) and CAP_SYS_ADMIN. Run
 * foreground with: sudo ./kllm -f
 */
#define FUSE_USE_VERSION 31

#include <cuse_lowlevel.h>
#include <fuse_opt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stddef.h>
#include "engine.h"

static struct kllm_engine *g_engine;

struct fh_state {
	char  *prompt;
	size_t plen, pcap;
	char  *report;
	size_t rlen, rpos;   /* report length and stream position consumed */
	int    computed;
};

static void
fh_reset_request(struct fh_state *st)
{
	st->plen = 0;
	free(st->report);
	st->report = NULL;
	st->rlen = 0;
	st->rpos = 0;
	st->computed = 0;
}

static void
kllm_open(fuse_req_t req, struct fuse_file_info *fi)
{
	struct fh_state *st = calloc(1, sizeof(*st));
	if (!st) {
		fuse_reply_err(req, ENOMEM);
		return;
	}
	fi->fh = (uintptr_t)st;
	fi->nonseekable = 1;
	fi->direct_io = 1;
	fuse_reply_open(req, fi);
}

static void
kllm_release(fuse_req_t req, struct fuse_file_info *fi)
{
	struct fh_state *st = (struct fh_state *)(uintptr_t)fi->fh;
	if (st) {
		free(st->prompt);
		free(st->report);
		free(st);
	}
	fuse_reply_err(req, 0);
}

static void
kllm_write(fuse_req_t req, const char *buf, size_t size, off_t off,
	   struct fuse_file_info *fi)
{
	(void)off;
	struct fh_state *st = (struct fh_state *)(uintptr_t)fi->fh;

	if (st->computed)            /* a new write starts a fresh request */
		fh_reset_request(st);

	if (st->plen + size > st->pcap) {
		size_t ncap = st->pcap ? st->pcap * 2 : 256;
		while (ncap < st->plen + size)
			ncap *= 2;
		char *np = realloc(st->prompt, ncap);
		if (!np) {
			fuse_reply_err(req, ENOMEM);
			return;
		}
		st->prompt = np;
		st->pcap = ncap;
	}
	memcpy(st->prompt + st->plen, buf, size);
	st->plen += size;
	fuse_reply_write(req, size);
}

static void
compute_report(struct fh_state *st)
{
	struct kllm_run_stats s;
	int rc = kllm_engine_run(g_engine, st->prompt, st->plen, &s);

	char ns[33];
	kllm_engine_namespace_hex(g_engine, ns);

	unsigned pct = s.pages ? (100u * s.hits) / s.pages : 0;
	char buf[512];
	int n = snprintf(buf, sizeof(buf),
		"kLLM\n"
		"namespace : %s\n"
		"tokens    : %u\n"
		"pages     : %u (page_tokens=%u, block=%llu B)\n"
		"cache     : hits %u  misses %u  (%u%% hit)\n"
		"fetched   : %llu B\n"
		"computed  : %llu B\n"
		"store     : %zu blocks  (rc=%d)\n",
		ns, s.tokens, s.pages, KLLM_MODEL_TINY.page_tokens,
		(unsigned long long)kllm_engine_block_bytes(g_engine),
		s.hits, s.misses, pct,
		(unsigned long long)s.bytes_fetched,
		(unsigned long long)s.bytes_computed,
		kllm_engine_block_count(g_engine), rc);

	st->report = malloc((size_t)n + 1);
	if (st->report) {
		memcpy(st->report, buf, (size_t)n + 1);
		st->rlen = (size_t)n;
	}
	st->computed = 1;
}

static void
kllm_read(fuse_req_t req, size_t size, off_t off, struct fuse_file_info *fi)
{
	(void)off;  /* device is nonseekable; the kernel passes off==0. Track our
		     * own stream position so the read terminates with EOF. */
	struct fh_state *st = (struct fh_state *)(uintptr_t)fi->fh;

	if (!st->computed)
		compute_report(st);

	if (!st->report) {
		fuse_reply_err(req, ENOMEM);
		return;
	}
	if (st->rpos >= st->rlen) {
		fuse_reply_buf(req, NULL, 0);  /* EOF: whole report delivered */
		return;
	}
	size_t avail = st->rlen - st->rpos;
	size_t n = size < avail ? size : avail;
	fuse_reply_buf(req, st->report + st->rpos, n);
	st->rpos += n;
}

static const struct cuse_lowlevel_ops kllm_clop = {
	.open    = kllm_open,
	.read    = kllm_read,
	.write   = kllm_write,
	.release = kllm_release,
};

int
main(int argc, char **argv)
{
	g_engine = kllm_engine_create(&KLLM_MODEL_TINY);
	if (!g_engine) {
		fprintf(stderr, "kllm: engine create failed\n");
		return 1;
	}

	const char *dev_info_argv[] = { "DEVNAME=kllm" };
	struct cuse_info ci = {
		.dev_info_argc = 1,
		.dev_info_argv = dev_info_argv,
		.flags = CUSE_UNRESTRICTED_IOCTL,
	};

	fprintf(stderr, "kllm: serving /dev/kllm (needs `modprobe cuse`, run as root)\n");
	int rc = cuse_lowlevel_main(argc, argv, &ci, &kllm_clop, NULL);

	kllm_engine_destroy(g_engine);
	return rc;
}
