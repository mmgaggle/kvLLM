/* SPDX-License-Identifier: BSD-3-Clause */
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "kvstore.h"

#define KVSTORE_BUCKETS 4096u  /* power of two */

struct kv_entry {
	struct kllm_key   key;
	void             *val;
	uint32_t          len;
	struct kv_entry  *next;
};

struct kllm_kvstore {
	struct kv_entry *buckets[KVSTORE_BUCKETS];
	size_t           count;
};

static inline uint32_t
bucket_of(struct kllm_key key)
{
	/* Key bytes are already a hash; fold the first 4 bytes into the index. */
	uint32_t h = ((uint32_t)key.b[0] << 24) | ((uint32_t)key.b[1] << 16) |
		     ((uint32_t)key.b[2] << 8)  |  (uint32_t)key.b[3];
	return h & (KVSTORE_BUCKETS - 1);
}

struct kllm_kvstore *
kllm_kvstore_create(void)
{
	return calloc(1, sizeof(struct kllm_kvstore));
}

void
kllm_kvstore_destroy(struct kllm_kvstore *ks)
{
	if (!ks)
		return;
	for (size_t i = 0; i < KVSTORE_BUCKETS; i++) {
		struct kv_entry *e = ks->buckets[i];
		while (e) {
			struct kv_entry *next = e->next;
			free(e->val);
			free(e);
			e = next;
		}
	}
	free(ks);
}

static struct kv_entry *
find(struct kllm_kvstore *ks, struct kllm_key key)
{
	for (struct kv_entry *e = ks->buckets[bucket_of(key)]; e; e = e->next)
		if (kllm_key_eq(e->key, key))
			return e;
	return NULL;
}

int
kllm_kvstore_exist(struct kllm_kvstore *ks, struct kllm_key key)
{
	return find(ks, key) != NULL;
}

int
kllm_kvstore_store(struct kllm_kvstore *ks, struct kllm_key key,
		   const void *val, uint32_t len)
{
	struct kv_entry *e = find(ks, key);

	if (e) {
		/* Overwrite. Content addressing means value should be identical,
		 * but honour the store verb regardless. */
		void *nv = malloc(len);
		if (!nv)
			return -ENOMEM;
		memcpy(nv, val, len);
		free(e->val);
		e->val = nv;
		e->len = len;
		return 0;
	}

	e = malloc(sizeof(*e));
	if (!e)
		return -ENOMEM;
	e->val = malloc(len);
	if (!e->val) {
		free(e);
		return -ENOMEM;
	}
	memcpy(e->val, val, len);
	e->key = key;
	e->len = len;

	uint32_t b = bucket_of(key);
	e->next = ks->buckets[b];
	ks->buckets[b] = e;
	ks->count++;
	return 0;
}

int
kllm_kvstore_retrieve(struct kllm_kvstore *ks, struct kllm_key key,
		      void *buf, uint32_t buf_len, uint32_t *out_len)
{
	struct kv_entry *e = find(ks, key);
	if (!e)
		return -ENOENT;

	uint32_t copy = e->len < buf_len ? e->len : buf_len;
	memcpy(buf, e->val, copy);
	if (out_len)
		*out_len = e->len;
	return 0;
}

size_t
kllm_kvstore_count(struct kllm_kvstore *ks)
{
	return ks->count;
}
