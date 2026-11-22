import argparse
import ctypes
import pathlib

import numpy as np

from lanescope.cuda import nvrtc
from lanescope.cuda.driver import Driver
from lanescope.cuda.toolchain import find

SOURCE = pathlib.Path(__file__).resolve().with_suffix(".cu")
KERNELS = ("histogram", "transpose", "tree_sum", "scale")
WIDTH = 128
HISTOGRAM_VALUES = 1 << 16


def main() -> None:
    """Launch a transpose, a histogram, a tree sum, and a scale, each with its kind of conflict."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--repeats", type=int, default=1, help="times to launch the four kernels")
    parser.add_argument("--blocks", type=int, default=8, help="CTAs of the histogram and the sums")
    args = parser.parse_args()
    driver = Driver()
    library = find().nvrtc
    if library is None:
        raise SystemExit("NVRTC was not found. Run pip install 'lanescope[gpu]' to add it.")
    # The line tables let the recording name the source line of each access.
    options = [f"--gpu-architecture={driver.arch()}", "-lineinfo"]
    program = nvrtc.compile_cuda(SOURCE.read_text(), str(SOURCE), options, library)
    module = driver.load(program.cubin)
    kernels = {name: driver.function(module, name) for name in KERNELS}
    rng = np.random.default_rng(12)
    floats = driver.alloc(WIDTH * WIDTH * 4)
    results = driver.alloc(WIDTH * WIDTH * 4)
    driver.upload(floats, rng.random(WIDTH * WIDTH, dtype=np.float32).tobytes())
    values = driver.alloc(HISTOGRAM_VALUES * 4)
    bins = driver.alloc(1024 * 4)
    for _ in range(args.repeats):
        for _ in range(3):
            data = rng.integers(0, 1 << 20, HISTOGRAM_VALUES, dtype=np.uint32)
            driver.upload(values, data.tobytes())
            count = ctypes.c_int(HISTOGRAM_VALUES)
            arguments = [ctypes.c_uint64(values), ctypes.c_uint64(bins), count]
            driver.launch(kernels["histogram"], (args.blocks, 1, 1), (256, 1, 1), arguments)
        for _ in range(2):
            arguments = [ctypes.c_uint64(floats), ctypes.c_uint64(results), ctypes.c_int(WIDTH)]
            grid = (WIDTH // 32, WIDTH // 32, 1)
            driver.launch(kernels["transpose"], grid, (32, 32, 1), arguments)
        for _ in range(2):
            arguments = [ctypes.c_uint64(floats), ctypes.c_uint64(results)]
            driver.launch(kernels["tree_sum"], (args.blocks, 1, 1), (256, 1, 1), arguments)
        arguments = [ctypes.c_uint64(floats), ctypes.c_uint64(results), ctypes.c_float(2.0)]
        driver.launch(kernels["scale"], (args.blocks, 1, 1), (256, 1, 1), arguments)
    driver.close()
    print(f"launched {8 * args.repeats} kernels")


if __name__ == "__main__":
    main()
