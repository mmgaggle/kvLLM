/* SPDX-License-Identifier: BSD-3-Clause */
#include "tokenizer.h"

size_t
kllm_tokenize(const void *text, size_t len, uint32_t *tokens, size_t max_tokens)
{
	const uint8_t *p = text;
	size_t n = len < max_tokens ? len : max_tokens;

	for (size_t i = 0; i < n; i++)
		tokens[i] = p[i];

	return n;
}
