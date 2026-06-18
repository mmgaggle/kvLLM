/* SPDX-License-Identifier: BSD-3-Clause */
/* Granite generate driver: reads prompt token ids on stdin, prints generated ids. */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include "granite.h"

int
main(int argc, char **argv)
{
	const char *wpath = argc > 1 ? argv[1] : "models/granite/granite.weights";
	int n_new = argc > 2 ? atoi(argv[2]) : 40;
	float temp = argc > 3 ? (float)atof(argv[3]) : 0.0f;

	std::vector<uint32_t> toks;
	long id;
	while (scanf("%ld", &id) == 1)
		toks.push_back((uint32_t)id);
	if (toks.empty()) {
		fprintf(stderr, "granite_gen: no input token ids on stdin\n");
		return 1;
	}

	Granite *g = granite_load(wpath);
	if (!g)
		return 1;
	fprintf(stderr, "granite_gen: %d prompt tokens, gen %d, temp %.2f\n",
		(int)toks.size(), n_new, temp);

	GraniteKV *kv = granite_kv_create(g, (int)toks.size() + n_new + 8);
	std::vector<uint32_t> out;
	granite_generate_cached(g, kv, toks, n_new, out, temp, 1234ULL);
	granite_kv_free(kv);

	for (size_t i = 0; i < out.size(); i++)
		printf("%u%s", out[i], i + 1 < out.size() ? " " : "");
	printf("\n");
	granite_free(g);
	return 0;
}
