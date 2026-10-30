# LaneScope

[![Tests](https://github.com/aalramadan/lanescope/actions/workflows/tests.yml/badge.svg)](https://github.com/aalramadan/lanescope/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Checked on an RTX 4060](https://img.shields.io/badge/checked%20on-RTX%204060-76b900)](#accuracy)

LaneScope is a profiler for shared-memory bank conflicts in CUDA kernels. It records the memory address of every thread in each warp, shows which threads collide in the same bank, and predicts the padding or XOR swizzle that removes each conflict before any code changes.

## Installation

LaneScope requires Python 3.10 or later. The analysis tools and the GUI run without a GPU.

```bash
git clone https://github.com/aalramadan/lanescope.git
cd lanescope
python3 -m venv .venv && . .venv/bin/activate
pip install .            # analysis, command line, and GUI
pip install ".[gpu]"     # recording on an NVIDIA GPU
```

Recording needs the setup listed under [Requirements](#requirements). On Ubuntu, `sudo apt install python3-venv g++` installs the two system packages.

## Quick Start

```bash
lanescope gui --demo
```

The command serves the GUI at http://localhost:8765 with three demo kernels. The first is the tiled matrix transpose of the CUDA samples, which has a 32-way bank conflict. Each step of the swizzle control halves the conflict, from 32 wavefronts per warp to one.

![The swizzle control of LaneScope reducing a 32-way bank conflict to none](docs/demo.gif)

## Features

| Feature | Description |
|---|---|
| Lane view | Shows each shared and global memory instruction lane by lane, beside its source line |
| Hardware counts | Reports the wavefronts and sectors that Nsight Compute reports, for loads, stores, `ldmatrix`, and `cp.async` |
| Layout search | Ranks the paddings and XOR swizzles of each buffer by the wavefronts each one removes |
| Code output | Writes the chosen layout as a byte offset, an element index, a 2D array index, or a CuTe `Swizzle<B, M, S>` |
| No rebuild | Records any CUDA program through a Compute Sanitizer tool, with no change to its build, including Triton kernels that PyTorch compiles at run time |
| Validation | Compares every count with Nsight Compute, on its own bench or on any program |
| Three interfaces | Offers a browser GUI, a command line, and a Model Context Protocol (MCP) server for coding agents |

## Usage

### Analyze a Trace

`lanescope explain` lists the counters of every memory instruction, worst first. The example analyzes the demo trace of the tiled transpose.

```console
$ lanescope demo --out transpose.npz
$ lanescope explain transpose.npz
kernel transpose (synthetic), 512 warp-level accesses

pc       line  space  kind  width  requests  unit        achieved  ideal excessive
0x0120     10  shared load      4       128  wavefronts      4096    128      3968
0x00a0      5  global load      4       128  sectors          512    512         0
0x00b0      5  shared store     4       128  wavefronts       128    128         0
0x0130     10  global store     4       128  sectors          512    512         0
```

The shared load at line 10 takes 4096 wavefronts against an ideal of 128. Nsight Compute counts each pass of a shared-memory access as one wavefront.

### Inspect One Instruction

`lanescope lanes` shows where each lane of one warp lands.

```console
$ lanescope lanes transpose.npz --pc 0x120
pc 0x0120, shared load of 4 bytes, CTA 0, warp 0
32 wavefronts against an ideal of 1, 31 bank conflicts

phase 1, lanes 0-31, 32 wavefronts, worst bank 0
bank    00000000001111111111222222222233
        01234567890123456789012345678901
lane  0 A...............................  offset 0x0
lane  1 B...............................  offset 0x80
lane  2 C...............................  offset 0x100
...
lane 31 f...............................  offset 0xf80
```

The 32 lanes read words that lie 128 bytes apart. Every word falls in bank 0. The load therefore needs 32 passes.

### Find a Fix

`lanescope suggest` ranks the layouts of each buffer by the excess wavefronts that remain.

```console
$ lanescope suggest transpose.npz
buffer 0x0+4096
XOR swizzle Swizzle<5,2,5> on byte offsets: excessive wavefronts 3968 -> 0, 0 extra bytes
pad each 128-byte row by 4 bytes: excessive wavefronts 3968 -> 0, 128 extra bytes
...
```

For a `float tile[32][32]`, the top-ranked swizzle changes one index.

```cpp
tile[r][c ^ (r & 0x1f)]   // CuTe: Swizzle<5, 0, 5>
```

`lanescope whatif` reports the counters for a single layout change. The commands `explain`, `whatif`, and `suggest` accept `--json` for use in scripts.

### Record a CUDA Program

```bash
lanescope record --out run1 -- ./my_app --size 4096
```

The command runs the program under the recorder and prints the path of each trace it writes. Compile the program with `-lineinfo` to map instructions to source lines.

| Option | Effect |
|---|---|
| `--kernel TEXT` | Records only the kernels whose names contain TEXT |
| `--ctas FIRST:COUNT` | Records a range of thread blocks, by default 64 blocks from the first |
| `--arch ARCH` | Sets the GPU architecture, such as `sm_89`, by default that of the local GPU |

Recording slows the program down. On large programs, limit it with `--kernel` and `--ctas`, then open the folder with `lanescope gui run1`.

### Record a Triton Kernel

The recorder also sees the kernels that Triton compiles inside a Python program. The example multiplies two matrices with a Triton kernel and needs PyTorch, which brings Triton.

```console
$ pip install torch
$ lanescope record --out triton --kernel matmul_kernel -- python examples/triton_matmul.py
$ lanescope explain triton/trace-0.npz
kernel matmul_kernel (sm_89), 10496 warp-level accesses

pc       line  space  kind  width  requests  unit        achieved  ideal excessive
0x0b60     15  shared store    16       512  wavefronts      3584   2048      1536
0x0b80     15  shared store    16       512  wavefronts      3584   2048      1536
...
```

Line 15 of the example loads a tile of `a` into shared memory with `cp.async`. Each row of that tile is 64 bytes long. Each wavefront of the copy writes one row. The copy therefore takes 3,584 wavefronts against an ideal of 2,048. With `--block-k 64`, each wavefront fills a full 128-byte line. LaneScope then reports no excess for the kernel.

### Use with Coding Agents

```bash
pip install ".[mcp]"
lanescope mcp
```

The command serves the analyses over MCP on standard input and output. An MCP client adds it with the command `lanescope` and the argument `mcp`.

## Accuracy

LaneScope computes its counts from the recorded addresses with a model of the hardware. The model is validated against Nsight Compute 2026.3 on an NVIDIA GeForce RTX 4060 Laptop GPU (sm_89). One run of the first `lanescope validate` command below gives the following counts.

| Test | Counts equal to Nsight Compute |
|---|---|
| 57 calibration launches of loads and stores | 114 of 114 |
| 52 calibration launches of `ldmatrix` and `cp.async` | 144 of 144 |
| 300 random loads and stores, kept out of the fit | 599 of 600 |
| 300 random `ldmatrix` and `cp.async` patterns, kept out of the fit | 872 of 872 |
| Transposes and matrix multiplies | 288 of 288 |
| Padded and swizzled kernels, predicted from the plain ones | 6 of 6 |

The second command checks any program in the same way. On the Triton example, with 32 and with 64 columns per step, all 304 counts match. The test suite checks the model against every pattern measured on that GPU, 1,355 loads and stores in `tests/data/rtx4060_counts.json` and 1,462 `ldmatrix` and `cp.async` patterns in `tests/data/rtx4060_instruction_counts.json`. The same comparison runs on any supported GPU.

```bash
lanescope get-ncu                                      # installs Nsight Compute for the local driver
lanescope validate --out check --random 300 --seed 4
lanescope validate --out triton-check --kernel matmul_kernel -- python examples/triton_matmul.py
```

## Performance

`lanescope bench time` times the transposes and a tensor-core matrix multiply that loads its tiles with `cp.async` and `ldmatrix`. Each runs with a plain layout and with the swizzle that `lanescope suggest` ranks first. The transpose also runs with padding. On the RTX 4060 Laptop GPU, it gives the following.

```console
$ lanescope bench build --arch sm_89 --out bench.cubin
$ lanescope bench time --cubin bench.cubin
8192 x 8192 float matrix, 50 launches per kernel
kernel                     ms     GB/s  speedup
transpose_tile          2.763    194.3    1.00x
transpose_padded        2.675    200.7    1.03x
transpose_swizzled      2.328    230.7    1.19x

4096 x 4096 x 4096 half-precision matrix multiply, 50 launches per kernel
kernel                     ms  TFLOP/s  speedup
matmul_plain            9.767     14.1    1.00x
matmul_swizzled         5.512     24.9    1.77x
```

In three runs, the swizzle makes the transpose 1.13 to 1.19 times faster and the matrix multiply 1.72 to 1.77 times faster. The command also checks that the two kernels of the matrix multiply give the same product.

## Requirements

| Component | Requirement |
|---|---|
| Operating system | Linux or Windows Subsystem for Linux 2 (WSL2) |
| Python | 3.10 or later |
| GPU for recording | An NVIDIA GPU with a driver for CUDA 12.4 or later |
| Compiler for recording | `g++` |
| Validated GPU | GeForce RTX 4060 Laptop GPU (Ada, sm_89) |
| Newer GPUs | sm_90 and later use rules from the NVIDIA developer forum |
| Triton example | PyTorch with CUDA, which brings Triton, and the Python development headers |

## How It Works

| Step | Module | Description |
|---|---|---|
| 1 | `recorder/` | Records the address of each active lane at every shared and global memory instruction, through Compute Sanitizer |
| 2 | `cuda/` | Maps each record to its SASS instruction and source line |
| 3 | `trace.py` | Stores the records of each kernel in a trace file |
| 4 | `model/` | Counts wavefronts and sectors from the addresses |
| 5 | `fixes/` | Applies each candidate layout to the addresses, counts again, and writes the code of the best |
| 6 | `gui/`, `cli.py`, `mcp_server.py` | Show the results in the browser, in the terminal, and to coding agents |
| 7 | `nsight/` | Validates the model against Nsight Compute |

The model counts one 4-byte word per bank per cycle in shared memory and aligned 32-byte sectors in global memory. Accesses of 8 and 16 bytes, `ldmatrix`, and `cp.async` follow rules measured on the GPU. All steps are deterministic and run locally.

## Limitations

- LaneScope models shared and global loads and stores, `ldmatrix`, and `cp.async`. It does not model atomic operations.
- The rules are calibrated on one GPU, the RTX 4060. LaneScope counts `stmatrix` with the rule of `ldmatrix`. That rule is not yet tested on sm_90 or newer, the only GPUs that run `stmatrix`.
- Twelve of the 1,355 measured loads and stores differ from Nsight Compute. All twelve are wide loads with inactive lanes.
- The rule for 16-byte `cp.async` is measured on copies that bypass L1, as `cp.async.cg` does.

## Troubleshooting

| Message | Cause | Resolution |
|---|---|---|
| `Nsight Compute has no access to the GPU counters` | The driver restricts the performance counters | On Windows, select Enable Developer Settings in the Desktop menu of the NVIDIA Control Panel, then allow all users under Developer, Manage GPU Performance Counters. On Linux, set the `nvidia` module option `NVreg_RestrictProfilingToAdminUsers=0` and reboot |
| `missing CUDA tools` | The `gpu` extra is not installed | Run `pip install ".[gpu]"` |
| `C++ compiler not found` | `g++` is missing | Run `sudo apt install g++` |
| `cannot load the CUDA driver` | No GPU is visible | Recording and validation need a GPU. Analysis and the GUI do not |
| `Python.h: No such file or directory` | Triton compiles a helper against the Python headers | Run `sudo apt install python3-dev` |

## Development

```bash
pip install -e ".[gpu,mcp]" pytest ruff pyright playwright
python -m playwright install chromium-headless-shell
pytest
```

## Acknowledgments

The swizzle notation follows CuTe from NVIDIA CUTLASS. The rules for H100 and newer GPUs come from threads on the NVIDIA developer forum about `LDS.128` wavefronts.

## License

LaneScope is released under the MIT License. See [LICENSE](LICENSE).
