# LaneScope

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Checked on an RTX 4060](https://img.shields.io/badge/checked%20on-RTX%204060-76b900)](#accuracy)

LaneScope is a bank conflict profiler for CUDA kernels. It records the address that every lane of every warp touches, shows you which threads collide in which shared memory bank, and predicts the padding or XOR swizzle that removes the conflict before you edit a line of code.

![LaneScope cutting a 32-way bank conflict to none as the swizzle widens, one bit at a time](docs/demo.gif)

## Features

- See each shared and global memory instruction lane by lane, next to its source line.
- Get the counts that Nsight Compute reports. On an RTX 4060, LaneScope matches it on 210 of 210 calibration counts and on 596 of 600 counts of random access patterns.
- Drag a padding or swizzle slider and watch the wavefronts of the whole kernel change.
- Copy the fix as a byte offset, an element index, a 2D array index, or a CuTe `Swizzle<B, M, S>`.
- Record any CUDA program without rebuilding it, through a Compute Sanitizer tool.
- Script every analysis from the command line or hand it to a coding agent over MCP.

## Install

LaneScope needs Python 3.10 or later. The analysis and the GUI run without a GPU.

```bash
git clone https://github.com/aalramadan/lanescope && cd lanescope
python3 -m venv .venv && . .venv/bin/activate
pip install -e .          # analysis and GUI
pip install -e ".[gpu]"   # recording on an NVIDIA GPU
```

On Ubuntu, `sudo apt install python3-venv g++` adds the two system packages it uses.

## Try it

```bash
lanescope gui --demo
```

Then open http://localhost:8765. The first of its three demo kernels is the tiled matrix transpose of the CUDA samples.

## A 32-way bank conflict, step by step

The command line gives the same analyses as the GUI. Start with the counters of every memory instruction, worst first.

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

The shared load on line 10 takes 32 times its ideal. One warp shows why.

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

The 32 lanes read 32 different words, 128 bytes apart. Every one of these words sits in bank 0. A bank serves one word per cycle. The load therefore waits for the last of the 32. `suggest` tries the paddings and swizzles of the buffer and ranks them by the excess left.

```console
$ lanescope suggest transpose.npz
buffer 0x0+4096
XOR swizzle Swizzle<5,2,5> on byte offsets: excessive wavefronts 3968 -> 0, 0 extra bytes
pad each 128-byte row by 4 bytes: excessive wavefronts 3968 -> 0, 128 extra bytes
...
```

For a `float tile[32][32]`, the swizzle comes down to one changed index.

```cpp
tile[r][c ^ (r & 0x1f)]   // CuTe: Swizzle<5, 0, 5>
```

## Record your own kernels

Recording needs an NVIDIA GPU whose driver supports CUDA 12.4 or later, plus `g++`. It runs on Linux and under WSL2. Build your program with `-lineinfo` to see source lines.

```bash
lanescope record --out run1 -- ./my_app --size 4096
lanescope gui run1
```

`--kernel` records only the kernels whose name holds a given text. `--ctas` limits the recording to a range of thread blocks, since a patch runs before every memory instruction and slows the program down.

## Accuracy

LaneScope computes its counts from the recorded addresses with a model of the hardware. The model is checked against Nsight Compute 2026.3 on an RTX 4060 Laptop GPU (sm_89).

| Test | Counts equal to Nsight Compute |
|---|---|
| 60 calibration launches | 210 of 210 |
| 300 random access patterns, kept out of the fit | 596 of 600 |
| Padded and swizzled transpose, predicted from the plain one | 4 of 4 |

The tests check the model against all 1,055 patterns measured on that GPU, in `tests/data/rtx4060_counts.json`. These two commands run the same check on your GPU. Under WSL2, Nsight Compute also needs the NVIDIA Control Panel setting that opens the GPU performance counters.

```bash
lanescope get-ncu                                      # Nsight Compute for your driver, no root
lanescope validate --out check --random 300 --seed 3
```

## How it works

1. A Compute Sanitizer tool patches every shared and global memory instruction. Each warp-level access becomes one record with the address of every active lane.
2. nvdisasm maps each record to its SASS instruction and its source line.
3. The model counts wavefronts and sectors. Shared memory serves one 4-byte word per bank per cycle. Accesses of 8 and 16 bytes run in groups of 16 and 8 lanes. Loads merge these groups in the cases measured on the GPU. Global memory moves aligned 32-byte sectors.
4. A layout change moves every recorded address inside the chosen buffer. The model then counts again.

The swizzle notation follows CuTe. The wide-access rules for H100 and newer GPUs come from the threads of the NVIDIA developer forum on `LDS.128` wavefronts.

## Limitations

- LaneScope covers shared and global loads and stores. It does not model `ldmatrix` or atomics. `cp.async` is not calibrated yet.
- The rules come from an RTX 4060. For H100 and newer GPUs, LaneScope uses the counts that users posted on the NVIDIA forum.
- Eleven of the 1,055 measured patterns still differ from Nsight Compute. All of them are wide loads with inactive lanes.

## Troubleshooting

- `Nsight Compute has no access to the GPU counters` means the driver keeps the performance counters closed. In the NVIDIA Control Panel of Windows, tick Enable Developer Settings in the Desktop menu. Then open Developer, Manage GPU Performance Counters, and allow access to all users. On Linux, set the `NVreg_RestrictProfilingToAdminUsers=0` option of the `nvidia` kernel module and reboot.
- `missing CUDA tools` means the `gpu` extra is not installed. Run `pip install -e ".[gpu]"`.
- `C++ compiler not found` means the recorder cannot be built. Run `sudo apt install g++`.
- `cannot load the CUDA driver` means no GPU is visible. Recording and `validate` need one, while the GUI and the analysis commands run without it.

## Development

```bash
pip install -e ".[gpu,mcp]" pytest ruff pyright playwright
python -m playwright install chromium-headless-shell
pytest
```

## License

MIT
