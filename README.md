# LaneScope

[![Tests](https://github.com/aalramadan/LaneScope/actions/workflows/tests.yml/badge.svg)](https://github.com/aalramadan/LaneScope/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Find and fix shared-memory bank conflicts in CUDA kernels.

![The swizzle control of LaneScope removing the bank conflicts of a matrix transpose and of a matrix multiply tile](docs/demo.gif)

LaneScope records the memory accesses of a running CUDA program, shows which threads collide in shared memory, and ranks the padding or swizzle that removes each conflict. The demo also runs in the browser at https://aalramadan.github.io/LaneScope/ with no install.

## Highlights

| Feature | Description |
|---|---|
| Thread view | Every shared and global memory access, thread by thread, beside its source line |
| Hardware counts | The wavefronts and sectors that Nsight Compute reports |
| Fix search | Paddings and swizzles ranked by the conflicts each removes, written as code |
| No rebuild | Any CUDA program, including the Triton kernels that PyTorch compiles |
| Three interfaces | A browser GUI, a command line, and an MCP server for coding agents |

## Installation

```bash
git clone https://github.com/aalramadan/LaneScope.git
cd LaneScope
pip install .            # analysis, command line, and GUI
pip install ".[gpu]"     # recording on an NVIDIA GPU
```

## Quick Start

```bash
lanescope gui --demo                          # the GUI with four demo kernels
lanescope record --out run1 -- ./my_app       # record a CUDA program
lanescope gui run1                            # explore the recording
lanescope suggest run1/trace-0.npz            # rank the fixes in the terminal
```

## Commands

| Command | Description |
|---|---|
| `record` | Records the memory accesses of a CUDA or Python program |
| `gui` | Opens the browser GUI |
| `explain` | Lists the memory instructions of a trace, worst first |
| `lanes` | Shows where each thread of a warp lands |
| `suggest` | Ranks the paddings and swizzles of each buffer |
| `validate` | Compares the counts with Nsight Compute |
| `mcp` | Serves the analyses to coding agents |

Run `lanescope <command> --help` for the options of each command.

## Requirements

- Linux or WSL2 with Python 3.10 or later
- For recording, `g++` and an NVIDIA GPU with a driver for CUDA 12.4 or later
- Validated on an RTX 4060 Laptop GPU

## How It Works

| Step | Module | Description |
|---|---|---|
| 1 | `recorder/` | Records the address of each thread at every memory access, through Compute Sanitizer |
| 2 | `model/` | Counts wavefronts and sectors from the addresses |
| 3 | `fixes/` | Tries paddings and swizzles on the recorded addresses and ranks them |
| 4 | `gui/`, `cli.py`, `mcp_server.py` | Show the results |
| 5 | `nsight/` | Checks the counts against Nsight Compute |

## Development

```bash
pip install -e ".[gpu,mcp]" pytest ruff pyright playwright
python -m playwright install chromium-headless-shell
pytest
```

## Acknowledgments

The swizzle notation follows CuTe from NVIDIA CUTLASS. The rules for H100 and newer GPUs come from the NVIDIA developer forum.

## License

LaneScope is released under the MIT License. See [LICENSE](LICENSE).
