# tq_check

Checks the C TurboQuant codec in `src/hnswtq.c` against
[Ginger](https://github.com/awitten1/ginger)'s TurboQuant, which it is ported
from.

- **`tq_check`** links Ginger and compares, on random vectors for 15
  dimensions (r = 1 to 31 blocks), two seeds, every extra-bit width and both
  metrics: the random signs and rotation matrix, rotated vectors, 4-bit codes,
  extra bits, norms and scaling factors, int8 and float queries, and 4-bit and
  4+k bit distances against Ginger's scalar reference scorers. It ends with
  recall@10 of 4-bit search reranked by 4+k bits on random data.
- **`tq_selftest`** needs only the codec. It checks the dot-product kernel
  (AVX2 when built with `-mavx2`) against a plain scalar one, and the 4+8 bit
  re-rank kernels (AVX2, and AVX-512 when the build has it) against the scalar
  loop, within float rounding. It prints a checksum of the codec's outputs on
  fixed inputs, from the scalar loop, which should be the same on every
  architecture.
- **`gen_codebooks`** writes `src/hnswtq_codebooks.h` from Ginger's codebook
  headers. Ginger computes several tables at compile time (the 4+k boundaries
  and the piecewise-cubic 4+6 and 4+8 bit codebooks), so they are generated
  rather than copied.

## Results

At commit `13c0929` of Ginger:

- Signs: identical. Rotation matrix: identical for r = 1 and 3; within about
  2e-6 for other r (`logf()` and Eigen's vectorized reductions round
  differently).
- 4-bit codes: identical for every dimension, k and seed.
- Extra bits: identical, except a few bytes per 100k dimensions at k = 8. The
  norm ||x|| is summed in double rather than in Eigen's float order, so a
  coordinate within an ulp of one of the fine 8-bit boundaries can land on the
  other side. Ginger itself varies the same way across SIMD targets.
- Distances: within about 4e-7 of Ginger's reference scorers, relative to the
  magnitude of the terms. For inner product, Ginger's dataset keeps the 4-bit
  scaling factor as a float16; the codec keeps a float, so the check rounds it
  first.
- `tq_selftest` gives the same checksum on arm64, x86-64 and x86-64 with AVX2
  and FMA. Built with pgvector's fast-math flags, it does not, so the
  extension must compile `hnswtq.c` without them.

## Build

`tq_check` needs Ginger, a C++20 compiler and CMake 3.24:

```sh
git submodule update --init third_party/ginger
cmake -S tools/tq_check -B tools/tq_check/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build tools/tq_check/build --target tq_check
tools/tq_check/build/tq_check
```

Ginger fetches its dependencies on the first configure. To reuse the sources
of an existing Ginger build instead, and to use a separate Ginger checkout of
the same commit:

```sh
D=/path/to/ginger/build/_deps
cmake -S tools/tq_check -B tools/tq_check/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGINGER_SRC=/path/to/ginger \
  -DFETCHCONTENT_SOURCE_DIR_ABSEIL-CPP=$D/abseil-cpp-src -DFETCHCONTENT_SOURCE_DIR_EIGEN=$D/eigen-src \
  -DFETCHCONTENT_SOURCE_DIR_HIGHWAY=$D/highway-src -DFETCHCONTENT_SOURCE_DIR_KAMINPAR=$D/kaminpar-src \
  -DFETCHCONTENT_SOURCE_DIR_ONETBB=$D/onetbb-src -DFETCHCONTENT_SOURCE_DIR_PARLAYLIB=$D/parlaylib-src \
  -DFETCHCONTENT_SOURCE_DIR_PROTOBUF=$D/protobuf-src
```

`tq_selftest` and `gen_codebooks`:

```sh
cc -std=c11 -O2 -ffp-contract=off -Isrc src/hnswtq.c tools/tq_check/tq_selftest.c -lm -o tq_selftest
cc -std=c11 -O2 -ffp-contract=off -mavx2 -mfma -Isrc src/hnswtq.c tools/tq_check/tq_selftest.c -lm -o tq_selftest_avx2

c++ -std=c++20 -I third_party/ginger tools/tq_check/gen_codebooks.cc -o gen_codebooks
./gen_codebooks > src/hnswtq_codebooks.h
```
