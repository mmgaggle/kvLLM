/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * Content hashing for kLLM block keys.
 *
 * A KV-cache block is content-addressed by a Merkle hash over its token prefix:
 *
 *     block_key[0] = H(namespace_seed ‖ tokens_of_block_0)
 *     block_key[i] = H(block_key[i-1] ‖ tokens_of_block_i)
 *
 * Identical prefixes therefore chain to identical keys and share the same
 * physical KV block — across requests and across nodes.
 *
 * This uses FNV-1a-128 truncated into a 16-byte key, which fits the stock
 * NVMe KV 1–16B key limit and is fine for a single-node hit-rate prototype.
 * A future version swaps this for BLAKE3-256 (32-byte key, vendor-extended) for
 * collision-free content addressing.
 */
#ifndef KLLM_CHASH_H
#define KLLM_CHASH_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 128-bit content key. A future version widens to 32 (BLAKE3-256). */
#define KLLM_KEY_LEN 16

struct kllm_key {
	uint8_t b[KLLM_KEY_LEN];
};

/* Hash arbitrary bytes into a content key. */
struct kllm_key kllm_chash(const void *data, size_t len);

/* Merkle step: key = H(parent.b ‖ data). Chains a block onto its prefix. */
struct kllm_key kllm_chash_chain(struct kllm_key parent, const void *data, size_t len);

/* Hex-format a key. out must hold at least 2*KLLM_KEY_LEN + 1 bytes. */
void kllm_key_hex(struct kllm_key k, char *out);

/* 1 if equal, 0 otherwise. */
int kllm_key_eq(struct kllm_key a, struct kllm_key b);

#ifdef __cplusplus
}
#endif

#endif /* KLLM_CHASH_H */
