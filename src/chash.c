/* SPDX-License-Identifier: BSD-3-Clause */
#include "chash.h"

/*
 * FNV-1a over 128 bits, computed with unsigned __int128. This is a placeholder
 * for BLAKE3-256; it is well-distributed enough to demonstrate
 * content-addressed prefix sharing at prototype scale, but is NOT
 * collision-resistant and must not gate correctness in production.
 */
#define FNV128_OFFSET ((((unsigned __int128)0x6c62272e07bb0142ULL) << 64) | 0x62b821756295c58dULL)
#define FNV128_PRIME  ((((unsigned __int128)0x0000000001000000ULL) << 64) | 0x000000000000013bULL)

static unsigned __int128
fnv1a_fold(unsigned __int128 h, const uint8_t *p, size_t len)
{
	for (size_t i = 0; i < len; i++) {
		h ^= (unsigned __int128)p[i];
		h *= FNV128_PRIME;
	}
	return h;
}

static struct kllm_key
key_from_state(unsigned __int128 h)
{
	struct kllm_key k;
	/* Big-endian: most-significant byte first. */
	for (int i = KLLM_KEY_LEN - 1; i >= 0; i--) {
		k.b[i] = (uint8_t)(h & 0xff);
		h >>= 8;
	}
	return k;
}

struct kllm_key
kllm_chash(const void *data, size_t len)
{
	unsigned __int128 h = fnv1a_fold(FNV128_OFFSET, data, len);
	return key_from_state(h);
}

struct kllm_key
kllm_chash_chain(struct kllm_key parent, const void *data, size_t len)
{
	unsigned __int128 h = FNV128_OFFSET;
	h = fnv1a_fold(h, parent.b, KLLM_KEY_LEN);
	h = fnv1a_fold(h, data, len);
	return key_from_state(h);
}

void
kllm_key_hex(struct kllm_key k, char *out)
{
	static const char hex[] = "0123456789abcdef";
	for (int i = 0; i < KLLM_KEY_LEN; i++) {
		out[i * 2]     = hex[k.b[i] >> 4];
		out[i * 2 + 1] = hex[k.b[i] & 0xf];
	}
	out[KLLM_KEY_LEN * 2] = '\0';
}

int
kllm_key_eq(struct kllm_key a, struct kllm_key b)
{
	for (int i = 0; i < KLLM_KEY_LEN; i++)
		if (a.b[i] != b.b[i])
			return 0;
	return 1;
}
