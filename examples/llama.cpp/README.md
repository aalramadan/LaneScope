# Example: the lightning indexer of llama.cpp

LaneScope found two bank conflicts in `lightning_indexer_kernel_mma`, the CUDA kernel behind the `LIGHTNING_INDEXER` op of llama.cpp. This op scores the earlier tokens for models that run attention on their top-scored tokens only, a design that DeepSeek V3.2 introduced and nine model implementations of llama.cpp now use. A small fix in [`lightning-indexer.patch`](lightning-indexer.patch) removes every conflict of the kernel and makes it **4.6% to 6.9% faster** on an RTX 5070 Laptop GPU.

The [live demo](https://aalramadan.github.io/LaneScope/) opens the recordings of the kernel before and after the fix.

Every result below comes from llama.cpp at commit `4b77dee0524c64b8471869232e2958bd9589607d` of 2026-11-05, built with CUDA 13.5 and timed on an RTX 5070 Laptop GPU with the driver for CUDA 13.5.

## 🔍 Record the Kernel

The commands below check out that commit, build llama.cpp with line information, and record one test case of the indexer.

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 4b77dee0524c64b8471869232e2958bd9589607d
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_FLAGS=-lineinfo
cmake --build build -j 8 --target test-backend-ops
lanescope record --out indexer --kernel lightning_indexer_kernel_mma -- \
    build/bin/test-backend-ops -o LIGHTNING_INDEXER -b CUDA0 -p 'nh=64,kv=256,nb=512,ns=1,nm=1,type_K=f16'
lanescope explain indexer/trace-0.npz
```

```text
pc       line  space  kind  width  requests  unit        achieved  ideal excessive
0x1210    147  shared store     8       512  wavefronts      4096   1024      3072
0x07f0    109  shared store     4      1024  wavefronts      2048   1024      1024
0x13b0    158  shared store     2       768  wavefronts      1536    768       768
```

## 💥 What Collides

- **Line 147**, `qk_chunk[head_local][kv_local] = Dqk.x[l];`, takes 4 times the ideal wavefronts. Each row of `qk_chunk` holds 32 floats. Every row therefore starts at bank 0. The result tile of the tensor cores puts four rows in each half-warp. These four rows land on the same eight banks.
- **Lines 108 to 111 and 158 to 161** write four halves per thread as separate 2-byte stores. The threads sit 8 bytes apart. Threads 0 and 16 therefore reach the same bank with different words.

## 🛠️ The Fix

1. Each row of `qk_chunk` grows by 8 floats, through `K_VECS_PADDED = K_VECS_PER_BLOCK + 8`. The four rows of a half-warp then start 8 banks apart.
2. The four halves go to shared memory in one 8-byte store through `ggml_cuda_memcpy_1<8>`, as the key tile of the same kernel already does.

The commands below apply the fix in the same checkout, then test and time the kernel.

```bash
git apply path/to/LaneScope/examples/llama.cpp/lightning-indexer.patch
cmake --build build -j 8 --target test-backend-ops
build/bin/test-backend-ops -o LIGHTNING_INDEXER -b CUDA0
build/bin/test-backend-ops perf -o LIGHTNING_INDEXER -b CUDA0 \
    -p 'nh=64,kv=(4096|65536),nb=(512|2048),ns=1,nm=1,type_K=(f16|q8_0)$'
```

After the fix, `lanescope explain` lists no instruction above its ideal. All 192 `LIGHTNING_INDEXER` tests of `test-backend-ops` pass.

## 📊 Results

Nsight Compute counts **3,588,096 shared wavefronts** for the recorded case after the fix, the ideal count, compared with 5,292,032 before.

Median time per run with 64 index heads, five alternating runs of each build:

| Keys | Tokens | Cache type | Before (µs) | After (µs) | Change |
|---|---|---|---|---|---|
| 4,096 | 512 | f16 | 4,069.8 | 3,826.0 | -6.0% |
| 4,096 | 512 | q8_0 | 4,314.7 | 4,055.1 | -6.0% |
| 4,096 | 2,048 | f16 | 17,447.3 | 16,435.2 | -5.8% |
| 4,096 | 2,048 | q8_0 | 18,280.5 | 17,271.1 | -5.5% |
| 65,536 | 512 | f16 | 69,195.9 | 64,448.3 | -6.9% |
| 65,536 | 512 | q8_0 | 71,371.5 | 66,740.5 | -6.5% |
| 65,536 | 2,048 | f16 | 265,612.5 | 250,743.5 | -5.6% |
| 65,536 | 2,048 | q8_0 | 276,955.2 | 264,167.2 | -4.6% |

llama.cpp is MIT licensed. The patch changes 15 lines of its file `ggml/src/ggml-cuda/lightning-indexer.cu`.
