# Default build is a fat binary: SASS for every arch from Maxwell to Hopper,
# plus PTX so that newer GPUs (Blackwell and later) still run via JIT. This
# means users don't need to know their compute capability or match a build to a
# card. Performance is irrelevant for this tool, so the extra build time and
# binary size cost nothing that matters.
#
#   make                    fat binary, works on anything
#   make ARCH=sm_86         single arch, builds much faster (development)
#   make ARCHES="sm_86 sm_89"   custom set
#
# If your CUDA SDK is older than a listed arch, drop it from ARCHES or pass
# ARCH= for a single target.

NVCC ?= nvcc

ARCHES ?= sm_52 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_89 sm_90
PTX    ?= compute_90

ifdef ARCH
  GENCODE := -arch=$(ARCH)
else
  GENCODE := $(foreach a,$(ARCHES),-gencode arch=compute_$(patsubst sm_%,%,$(a)),code=$(a)) \
             -gencode arch=$(PTX),code=$(PTX)
endif

# Maxwell and Pascal are deprecated in recent SDKs but still build and still
# run. Silence the warning rather than dropping support for older cards, which
# are exactly the ones most likely to have developed a bad cell.
FLAGS := -O3 $(GENCODE) -Wno-deprecated-gpu-targets

all: vramcheck vrampill vrampill2

vramcheck: src/vramcheck.cu
	$(NVCC) $(FLAGS) -o $@ $<

vrampill: src/vrampill.cu
	$(NVCC) $(FLAGS) -o $@ $<

# vrampill2 uses the driver API (VMM, file-descriptor export), so it links libcuda.
vrampill2: src/vrampill2.cu
	$(NVCC) $(FLAGS) -o $@ $< -lcuda

# Quick single-arch build for iterating.
dev:
	$(MAKE) ARCH=sm_86

clean:
	rm -f vramcheck vrampill vrampill2

.PHONY: all dev clean
