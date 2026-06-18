/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * In-memory content-addressed KV store.
 *
 * This deliberately mirrors the verb shape of SPDK's kvdev
 * (store / retrieve / exist, keyed by a short binary key) so that swapping it
 * for the real rados-nkvx kvdev over NVMe-oF is mechanical — the
 * engine calls these verbs, not a map. The only thing that changes later is the
 * backend and the (currently synchronous) completion model.
 */
#ifndef KLLM_KVSTORE_H
#define KLLM_KVSTORE_H

#include <stdint.h>
#include <stddef.h>
#include "chash.h"

#ifdef __cplusplus
extern "C" {
#endif

struct kllm_kvstore;

struct kllm_kvstore *kllm_kvstore_create(void);
void kllm_kvstore_destroy(struct kllm_kvstore *ks);

/* 1 if key is present, 0 otherwise. */
int kllm_kvstore_exist(struct kllm_kvstore *ks, struct kllm_key key);

/* Insert or overwrite. Copies value. Returns 0 or -ENOMEM. */
int kllm_kvstore_store(struct kllm_kvstore *ks, struct kllm_key key,
		       const void *val, uint32_t len);

/*
 * Retrieve into buf (up to buf_len bytes). On success sets *out_len to the true
 * stored length and returns 0; returns -ENOENT if absent. If buf_len < stored
 * length, copies buf_len bytes, sets *out_len to the true length, and still
 * returns 0 (cf. kvdev BUFFER_TOO_SMALL semantics — caller checks out_len).
 */
int kllm_kvstore_retrieve(struct kllm_kvstore *ks, struct kllm_key key,
			  void *buf, uint32_t buf_len, uint32_t *out_len);

/* Number of distinct keys held. */
size_t kllm_kvstore_count(struct kllm_kvstore *ks);

#ifdef __cplusplus
}
#endif

#endif /* KLLM_KVSTORE_H */
