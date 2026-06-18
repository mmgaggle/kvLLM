# kLLM — char-device head + content-addressed KV spine, and the GPU compute.
CC      ?= gcc
CFLAGS  ?= -O2 -g -std=c11 -Wall -Wextra
CFLAGS  += -Isrc -fPIC   # -fPIC so the C core can link into a HIP PIE

FUSE_CFLAGS := $(shell pkg-config --cflags fuse3)
FUSE_LIBS   := $(shell pkg-config --libs fuse3)

CORE_SRC := src/tokenizer.c src/chash.c src/kvstore.c src/engine.c
CORE_OBJ := $(CORE_SRC:.c=.o)

all: kllm-spine-test kllm

# Standalone spine test — no CUSE/kernel needed.
kllm-spine-test: $(CORE_OBJ) test/spine_test.o
	$(CC) $(CFLAGS) -o $@ $^

# /dev/kllm CUSE daemon. Needs fuse3; running needs `modprobe cuse` + root.
kllm: $(CORE_OBJ) src/cuse_dev.o
	$(CC) $(CFLAGS) -o $@ $^ $(FUSE_LIBS)

src/cuse_dev.o: src/cuse_dev.c
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

test: kllm-spine-test
	./kllm-spine-test

# ---- GPU components — require ROCm/hipcc. The default build stays CPU-only. ----
HIPCC     ?= hipcc
CXX       ?= g++
HIPCFLAGS ?= -O2 -std=c++17 -fPIC -Isrc -Igpu
HIPLDLIBS ?= -lamdhip64

gpu/kmodel_ref.o: gpu/kmodel_ref.cpp gpu/kmodel.h
	$(CXX) -O2 -std=c++17 -fPIC -Igpu -c -o $@ $<
gpu/kmodel_hip.o: gpu/kmodel_hip.hip gpu/kmodel.h
	$(HIPCC) $(HIPCFLAGS) -c -o $@ $<

# HIP forward vs CPU reference (numerical parity).
gpu-test: gpu/forward_test
	./gpu/forward_test
gpu/forward_test: gpu/kmodel_ref.o gpu/kmodel_hip.o gpu/forward_test.o
	$(HIPCC) $(HIPCFLAGS) -o $@ $^ $(HIPLDLIBS)
gpu/forward_test.o: gpu/forward_test.cpp gpu/kmodel.h
	$(CXX) -O2 -std=c++17 -fPIC -Igpu -c -o $@ $<

# Greedy decode + content-addressed KV (GPU compute + storage spine).
gpu-decode: gpu/decode
	./gpu/decode
gpu/decode: src/chash.o src/kvstore.o gpu/kmodel_ref.o gpu/kmodel_hip.o gpu/decode.o
	$(HIPCC) $(HIPCFLAGS) -o $@ $^ $(HIPLDLIBS)
gpu/decode.o: gpu/decode.cpp gpu/kmodel.h src/chash.h src/kvstore.h
	$(CXX) -O2 -std=c++17 -fPIC -Isrc -Igpu -c -o $@ $<

# Cache invariant: cached fp16 prefix continuation == full recompute.
gpu-cache-test: gpu/cache_test
	./gpu/cache_test
gpu/cache_test: gpu/kmodel_ref.o gpu/kmodel_hip.o gpu/cache_test.o
	$(HIPCC) $(HIPCFLAGS) -o $@ $^ $(HIPLDLIBS)
gpu/cache_test.o: gpu/cache_test.cpp gpu/kmodel.h
	$(CXX) -O2 -std=c++17 -fPIC -Igpu -c -o $@ $<

# Real GPT-2 (124M). `make gpt2` builds the generator; run via run_gpt2.py.
gpt2: gpu/gpt2_gen
gpu/gpt2_gen: gpu/gpt2.o gpu/gpt2_gen.o
	$(HIPCC) $(HIPCFLAGS) -o $@ $^ $(HIPLDLIBS)
gpu/gpt2.o: gpu/gpt2.hip gpu/gpt2.h
	$(HIPCC) $(HIPCFLAGS) -c -o $@ $<
gpu/gpt2_gen.o: gpu/gpt2_gen.cpp gpu/gpt2.h
	$(CXX) -O2 -std=c++17 -fPIC -Igpu -c -o $@ $<

# Real IBM Granite 4.0 nano. `make granite` builds the generator; run via run_granite.py.
granite: gpu/granite_gen
gpu/granite_gen: gpu/granite.o gpu/granite_gen.o
	$(HIPCC) $(HIPCFLAGS) -o $@ $^ $(HIPLDLIBS)
gpu/granite.o: gpu/granite.hip gpu/granite.h
	$(HIPCC) $(HIPCFLAGS) -c -o $@ $<
gpu/granite_gen.o: gpu/granite_gen.cpp gpu/granite.h
	$(CXX) -O2 -std=c++17 -fPIC -Igpu -c -o $@ $<

# /dev/kllm serving real GPT-2 with content-addressed KV (CUSE daemon).
serve: gpu/kllm-serve
gpu/kllm-serve: src/chash.o src/kvstore.o gpu/granite.o gpu/kllm_serve.o
	$(HIPCC) $(HIPCFLAGS) -o $@ $^ $(HIPLDLIBS) $(FUSE_LIBS)
gpu/kllm_serve.o: gpu/kllm_serve.cpp gpu/granite.h src/chash.h src/kvstore.h
	$(CXX) -O2 -std=c++17 -fPIC -Isrc -Igpu $(FUSE_CFLAGS) -c -o $@ $<

# Incremental KV cache: identical output to recompute, O(n) not O(n^2).
gpu-kvcache: gpu/kvcache_test
	./gpu/kvcache_test
gpu/kvcache_test: gpu/gpt2.o gpu/kvcache_test.o
	$(HIPCC) $(HIPCFLAGS) -o $@ $^ $(HIPLDLIBS)
gpu/kvcache_test.o: gpu/kvcache_test.cpp gpu/gpt2.h
	$(CXX) -O2 -std=c++17 -fPIC -Igpu -c -o $@ $<

# Warm-start: a cache hit loads the content-addressed prefix and skips prefill.
gpu-warmstart: gpu/warmstart_test
	./gpu/warmstart_test
gpu/warmstart_test: gpu/gpt2.o gpu/warmstart_test.o
	$(HIPCC) $(HIPCFLAGS) -o $@ $^ $(HIPLDLIBS)
gpu/warmstart_test.o: gpu/warmstart_test.cpp gpu/gpt2.h
	$(CXX) -O2 -std=c++17 -fPIC -Igpu -c -o $@ $<

clean:
	rm -f $(CORE_OBJ) src/cuse_dev.o test/spine_test.o kllm kllm-spine-test \
	      gpu/kmodel_ref.o gpu/kmodel_hip.o gpu/forward_test.o gpu/forward_test \
	      gpu/decode.o gpu/decode gpu/cache_test.o gpu/cache_test \
	      gpu/gpt2.o gpu/gpt2_gen.o gpu/gpt2_gen \
	      gpu/kvcache_test.o gpu/warmstart_test gpu/warmstart_test.o \
	      gpu/kllm_serve.o gpu/kllm-serve \
	      gpu/granite.o gpu/granite_gen.o gpu/granite_gen

.PHONY: all test gpu-test gpu-decode gpu-cache-test gpt2 granite serve gpu-kvcache gpu-warmstart clean
