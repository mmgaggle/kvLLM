/* SPDX-License-Identifier: BSD-3-Clause */
/* GPT-2 generate driver: reads prompt token ids on stdin, prints generated ids.
 * Tokenization/detokenization is done in userspace by gpu/run_gpt2.py. */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include "gpt2.h"

int
main(int argc, char **argv)
{
	const char *wpath = argc > 1 ? argv[1] : "models/gpt2/gpt2.weights";
	int n_new = argc > 2 ? atoi(argv[2]) : 20;

	std::vector<uint32_t> toks;
	long id;
	while (scanf("%ld", &id) == 1)
		toks.push_back((uint32_t)id);
	if (toks.empty()) {
		fprintf(stderr, "gpt2_gen: no input token ids on stdin\n");
		return 1;
	}

	Gpt2 *g = gpt2_load(wpath);
	if (!g)
		return 1;
	fprintf(stderr, "gpt2_gen: %d prompt tokens, generating %d...\n",
		(int)toks.size(), n_new);

	std::vector<uint32_t> out;
	gpt2_generate(g, toks, n_new, out);

	for (size_t i = 0; i < out.size(); i++)
		printf("%u%s", out[i], i + 1 < out.size() ? " " : "");
	printf("\n");
	gpt2_free(g);
	return 0;
}
