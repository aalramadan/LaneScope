import ctypes
import dataclasses
import pathlib
import random

import numpy as np

from lanescope.cuda import nvrtc
from lanescope.cuda.driver import Driver
from lanescope.cuda.toolchain import Toolchain, find
from lanescope.recorder.build import check_arch
from lanescope.trace import WARP_SIZE, Kind, Space

SHARED_BYTES = 32768
GLOBAL_BYTES = 1 << 20
ALL = 0xFFFFFFFF
TRANSPOSE_SIZE = 64
TRANSPOSE_KERNELS = ("transpose_tile", "transpose_padded", "transpose_swizzled")
LDMATRIX_KERNELS = ("ldmatrix_x1", "ldmatrix_x2", "ldmatrix_x4", "ldmatrix_x4_trans")
MATMUL_KERNELS = ("matmul_plain", "matmul_swizzled")
# Each block of 128 threads computes a 64 x 64 tile of the product. The calibration run
# multiplies 128 x 128 matrices, which takes four blocks.
MATMUL_TILE = 64
MATMUL_THREADS = 128
MATMUL_SIZE = 128
CP_ASYNC_WIDTHS = (4, 8, 16)
# The global buffer starts at zero, and the store cases write lane numbers below 64.
# No loaded word, nor the XOR of the words of one load, can therefore equal the sink.
SINK = 0xFFFFFFFF


@dataclasses.dataclass(frozen=True)
class Case:
    """One calibration launch: a lane pattern for one instruction."""

    name: str
    space: Space
    kind: Kind
    width: int
    offsets: tuple[int, ...]
    mask: int = ALL
    # The kernel of an ldmatrix or cp.async case, each of which runs an instruction of its own.
    instruction: str = ""

    @property
    def kernel(self) -> str:
        """Return the name of the kernel that runs this case."""
        if self.instruction:
            return self.instruction
        return f"{self.space.name.lower()}_{self.kind.name.lower()}_{self.width}"

    def addresses(self) -> tuple[int | None, ...]:
        """Return the offset of each lane, with None for lanes outside the mask."""
        return tuple(o if self.mask >> lane & 1 else None for lane, o in enumerate(self.offsets))


def _lanes(offset_of) -> tuple[int, ...]:
    return tuple(offset_of(lane) for lane in range(WARP_SIZE))


# LDS.128 patterns in int4 units with the wavefronts reported on the NVIDIA forum.
FORUM_LDS128 = (
    [0] * 8 + [1] * 8 + [2] * 8 + [3] * 8,
    [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9, 8, 9, 10, 11, 10, 11]
    + [12, 13, 12, 13, 14, 15, 14, 15],
    [0, 1, 0, 1, 2, 3, 2, 2, 4, 5, 4, 5, 6, 7, 6, 7, 0, 1, 0, 1, 2, 3, 2, 3]
    + [4, 5, 4, 5, 6, 7, 6, 7],
    [0, 1, 2, 3] * 8,
    list(range(8)) * 4,
    [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15] * 2,
    [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15] * 2,
    [0, 2, 0, 2, 4, 6, 4, 6, 8, 10, 8, 10, 12, 14, 12, 14, 16, 18, 16, 18, 20, 22, 20, 22]
    + [24, 26, 24, 26, 28, 30, 28, 30],
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] * 2,
)


def cases() -> list[Case]:
    """Return every calibration case in launch order."""
    shared, glob = Space.SHARED, Space.GLOBAL
    load, store = Kind.LOAD, Kind.STORE
    found = [
        Case(
            f"shared-load-4-stride-{4 * words}",
            shared,
            load,
            4,
            _lanes(lambda lane, w=words: 4 * w * lane),
        )
        for words in (1, 2, 3, 4, 8, 16, 32)
    ]
    found += [
        Case("shared-load-4-broadcast", shared, load, 4, (64,) * WARP_SIZE),
        Case("shared-load-4-multicast", shared, load, 4, _lanes(lambda lane: 4 * (lane // 4))),
        Case(
            "shared-load-4-permuted-banks",
            shared,
            load,
            4,
            _lanes(lambda lane: 4 * (5 * lane % 32)),
        ),
        Case(
            "shared-load-4-half-warp", shared, load, 4, _lanes(lambda lane: 128 * lane), 0x0000FFFF
        ),
        Case(
            "shared-load-4-even-lanes", shared, load, 4, _lanes(lambda lane: 128 * lane), 0x55555555
        ),
        Case("shared-load-1-contiguous", shared, load, 1, _lanes(lambda lane: lane)),
        Case("shared-load-1-stride-128", shared, load, 1, _lanes(lambda lane: 128 * lane)),
        Case("shared-load-2-contiguous", shared, load, 2, _lanes(lambda lane: 2 * lane)),
        Case("shared-load-2-stride-128", shared, load, 2, _lanes(lambda lane: 128 * lane)),
        Case("shared-load-8-contiguous", shared, load, 8, _lanes(lambda lane: 8 * lane)),
        Case("shared-load-8-stride-128", shared, load, 8, _lanes(lambda lane: 128 * lane)),
        Case("shared-load-8-stride-64", shared, load, 8, _lanes(lambda lane: 64 * lane)),
        Case("shared-load-8-pairs-aabb", shared, load, 8, _lanes(lambda lane: 8 * (lane // 2))),
        Case(
            "shared-load-8-pairs-abab",
            shared,
            load,
            8,
            _lanes(lambda lane: 8 * (2 * (lane // 4) + lane % 2)),
        ),  # fmt: skip
        Case(
            "shared-load-8-lanes-0-and-16",
            shared,
            load,
            8,
            _lanes(lambda lane: 8 * lane),
            0x00010001,
        ),
        Case("shared-load-16-contiguous", shared, load, 16, _lanes(lambda lane: 16 * lane)),
        Case("shared-load-16-stride-128", shared, load, 16, _lanes(lambda lane: 128 * lane)),
        Case(
            "shared-load-16-one-lane-per-quarter",
            shared,
            load,
            16,
            _lanes(lambda lane: 16 * lane),
            0x01010101,
        ),  # fmt: skip
        Case(
            "shared-load-16-pairs-partial-mask",
            shared,
            load,
            16,
            _lanes(lambda lane: 16 * (lane // 8)),
            0x03030303,
        ),  # fmt: skip
    ]
    found += [
        Case(f"shared-load-16-forum-{number}", shared, load, 16, tuple(16 * i for i in pattern))
        for number, pattern in enumerate(FORUM_LDS128, start=1)
    ]
    for width in (4, 8, 16):
        found += [
            Case(
                f"shared-store-{width}-contiguous",
                shared,
                store,
                width,
                _lanes(lambda lane, w=width: w * lane),
            ),
            Case(
                f"shared-store-{width}-stride-128",
                shared,
                store,
                width,
                _lanes(lambda lane: 128 * lane),
            ),
        ]
    found.append(
        Case("shared-store-16-forum-1", shared, store, 16, tuple(16 * i for i in FORUM_LDS128[0]))
    )
    found += [
        Case("global-load-4-coalesced", glob, load, 4, _lanes(lambda lane: 4 * lane)),
        Case("global-load-4-stride-8", glob, load, 4, _lanes(lambda lane: 8 * lane)),
        Case("global-load-4-offset-4", glob, load, 4, _lanes(lambda lane: 4 + 4 * lane)),
        Case("global-load-4-offset-96", glob, load, 4, _lanes(lambda lane: 96 + 4 * lane)),
        Case("global-load-4-column", glob, load, 4, _lanes(lambda lane: 4096 * lane)),
        Case("global-load-4-broadcast", glob, load, 4, (256,) * WARP_SIZE),
        Case("global-load-4-half-warp", glob, load, 4, _lanes(lambda lane: 4 * lane), 0x0000FFFF),
        Case("global-load-1-coalesced", glob, load, 1, _lanes(lambda lane: lane)),
        Case("global-load-2-coalesced", glob, load, 2, _lanes(lambda lane: 2 * lane)),
        Case("global-load-8-coalesced", glob, load, 8, _lanes(lambda lane: 8 * lane)),
        Case("global-load-16-coalesced", glob, load, 16, _lanes(lambda lane: 16 * lane)),
        Case("global-load-16-column", glob, load, 16, _lanes(lambda lane: 4096 * lane)),
        Case("global-store-4-coalesced", glob, store, 4, _lanes(lambda lane: 4 * lane)),
        Case("global-store-4-stride-8", glob, store, 4, _lanes(lambda lane: 8 * lane)),
        Case("global-store-4-column", glob, store, 4, _lanes(lambda lane: 4096 * lane)),
    ]
    for case in found:
        limit = SHARED_BYTES if case.space == Space.SHARED else GLOBAL_BYTES
        assert all(o % case.width == 0 and o + case.width <= limit for o in case.offsets), case.name
    return found


def instruction_cases() -> list[Case]:
    """Return every ldmatrix and cp.async case in launch order."""
    found = []
    for kernel, kind, width in _instruction_kernels():
        stem = kernel.replace("_", "-")
        found += [
            Case(f"{stem}-{pattern}", Space.SHARED, kind, width, offsets, instruction=kernel)
            for pattern, offsets in _instruction_patterns(width).items()
        ]
    return found


def _instruction_kernels() -> list[tuple[str, Kind, int]]:
    # ldmatrix gives each lane the address of one 16-byte row, and cp.async writes shared memory.
    found = [(kernel, Kind.LOAD, 16) for kernel in LDMATRIX_KERNELS]
    return found + [(f"cp_async_{width}", Kind.STORE, width) for width in CP_ASYNC_WIDTHS]


def _instruction_patterns(width: int) -> dict[str, tuple[int, ...]]:
    patterns = {
        "contiguous": _lanes(lambda lane: width * lane),
        "stride-128": _lanes(lambda lane: 128 * lane),
        "stride-64": _lanes(lambda lane: 64 * lane),
        "stride-32": _lanes(lambda lane: 32 * lane),
        "swizzled": _lanes(lambda lane: 128 * lane + 16 * (lane % 8)),
        "same-address": (0,) * WARP_SIZE,
    }
    if width == 16:
        patterns["stride-256"] = _lanes(lambda lane: 256 * lane)
        patterns["pairs"] = _lanes(lambda lane: 16 * (lane // 2))
    return patterns


def launch_groups(random_count: int = 0, seed: int = 0) -> list[tuple[str, list[Case]]]:
    """Return the cases of a calibration run in launch order, as groups with a description each."""
    groups = [
        ("calibration launches", cases()),
        ("ldmatrix and cp.async launches", instruction_cases()),
    ]
    if random_count:
        groups += [
            (f"random lane patterns of seed {seed}", random_cases(random_count, seed)),
            (
                f"random ldmatrix and cp.async patterns of seed {seed}",
                random_instruction_cases(random_count, seed),
            ),
        ]
    return groups


def launches(random_count: int = 0, seed: int = 0) -> list[Case]:
    """Return the cases of a calibration run, the fixed ones first and then the random ones."""
    return [case for _, group in launch_groups(random_count, seed) for case in group]


def random_cases(count: int, seed: int) -> list[Case]:
    """Return `count` random lane patterns for the calibration kernels, the same for one seed.

    Wide shared accesses come four times as often as the other shapes, since the
    rules that merge their lane groups are the hardest to get right.
    """
    rng = random.Random(seed)
    shapes = sorted({(case.space, case.kind, case.width) for case in cases()})
    weights = [4 if space == Space.SHARED and width in (8, 16) else 1 for space, _, width in shapes]
    found = []
    for index in range(count):
        space, kind, width = rng.choices(shapes, weights)[0]
        limit = (SHARED_BYTES if space == Space.SHARED else GLOBAL_BYTES) // width
        units = _random_units(rng, limit)
        name = f"random-{index}-{space.name.lower()}-{kind.name.lower()}-{width}"
        offsets = tuple(unit * width for unit in units)
        found.append(Case(name, space, kind, width, offsets, _random_mask(rng)))
    return found


def random_instruction_cases(count: int, seed: int) -> list[Case]:
    """Return `count` random ldmatrix and cp.async lane patterns, the same for one seed."""
    # A stream of its own leaves the measured random cases of the other kernels as they are.
    rng = random.Random(f"instructions-{seed}")
    kernels = _instruction_kernels()
    found = []
    for index in range(count):
        kernel, kind, width = rng.choice(kernels)
        offsets = tuple(unit * width for unit in _random_units(rng, SHARED_BYTES // width))
        name = f"random-{index}-{kernel.replace('_', '-')}"
        found.append(Case(name, Space.SHARED, kind, width, offsets, instruction=kernel))
    return found


def _random_mask(rng: random.Random) -> int:
    roll = rng.random()
    if roll < 0.55:
        return ALL
    if roll < 0.75:
        return rng.getrandbits(32) or 1
    step = rng.choice([2, 4, 8])
    phase = rng.randrange(step)
    return sum(1 << lane for lane in range(WARP_SIZE) if lane % step == phase) or 1


def _random_units(rng: random.Random, limit: int) -> list[int]:
    pattern = rng.choice(["random", "pairs1", "pairs2", "groups", "stride", "permuted", "halves"])
    span = rng.choice([2, 4, 8, 16, 32, 64, 128])
    if pattern == "random":
        units = [rng.randrange(span) for _ in range(WARP_SIZE)]
    elif pattern == "pairs1":
        base = [rng.randrange(span) for _ in range(16)]
        units = [base[lane // 2] for lane in range(WARP_SIZE)]
    elif pattern == "pairs2":
        base = [rng.randrange(span) for _ in range(16)]
        units = [base[(lane // 4) * 2 + lane % 2] for lane in range(WARP_SIZE)]
    elif pattern == "groups":
        size = rng.choice([2, 4, 8, 16])
        base = [rng.randrange(span) for _ in range(WARP_SIZE // size)]
        units = [base[lane // size] for lane in range(WARP_SIZE)]
    elif pattern == "stride":
        stride = rng.randrange(1, 10)
        start = rng.randrange(span)
        units = [start + stride * lane for lane in range(WARP_SIZE)]
    elif pattern == "permuted":
        units = list(range(WARP_SIZE))
        rng.shuffle(units)
    else:
        first = [rng.randrange(span) for _ in range(16)]
        units = first + (first if rng.random() < 0.5 else [rng.randrange(span) for _ in range(16)])
    # A changed lane breaks the pattern, as in the forum case whose pairs break once.
    if rng.random() < 0.3:
        units[rng.randrange(WARP_SIZE)] = rng.randrange(span)
    return [min(unit, limit - 1) for unit in units]


def kernel_source() -> str:
    """Return the CUDA source of the kernels that the cases and the transposes run."""
    shapes = sorted({(case.space, case.kind, case.width) for case in cases()})
    kernels = [_access_kernel(space, kind, width) for space, kind, width in shapes]
    kernels += [_ldmatrix_kernel(name) for name in LDMATRIX_KERNELS]
    kernels += [_cp_async_kernel(width) for width in CP_ASYNC_WIDTHS]
    return "\n".join([_PRELUDE, *kernels, _TRANSPOSES, _MATMULS])


def compile_kernels(
    arch: str, tools: Toolchain | None = None, source_path: pathlib.Path | None = None
) -> bytes:
    """Compile every kernel into one cubin for `arch` with NVRTC, with line information.

    With `source_path`, the source is written there first, and the line information
    names that file, which lets a trace of the run carry the source text.
    """
    check_arch(arch)
    tools = tools or find()
    tools.require("nvrtc")
    assert tools.nvrtc
    source = kernel_source()
    if source_path is not None:
        source_path.write_text(source)
    program = nvrtc.compile_cuda(
        source,
        str(source_path.resolve()) if source_path is not None else "lanescope_bench.cu",
        [f"--gpu-architecture={arch}", "-lineinfo"],
        tools.nvrtc,
    )
    assert program.cubin is not None
    return program.cubin


@dataclasses.dataclass(frozen=True)
class TransposeTime:
    """The time of one transpose kernel on an n by n float matrix."""

    kernel: str
    milliseconds: float
    size: int

    @property
    def gigabytes_per_second(self) -> float:
        """The bandwidth the kernel reaches, counting one read and one write of each element."""
        return 2 * 4 * self.size * self.size / (self.milliseconds * 1e-3) / 1e9


def time_transposes(cubin_path: pathlib.Path, size: int, repeats: int) -> list[TransposeTime]:
    """Time the three transposes on the local GPU, after one launch of each to warm up."""
    if size <= 0 or size % 32:
        raise ValueError(f"the matrix size {size} must be a positive multiple of 32")
    driver = Driver()
    module = driver.load(cubin_path.read_bytes())
    matrix_bytes = 4 * size * size
    source, target = driver.alloc(matrix_bytes), driver.alloc(matrix_bytes)
    grid, block = (size // 32, size // 32, 1), (32, 8, 1)
    arguments = [ctypes.c_uint64(target), ctypes.c_uint64(source), ctypes.c_int32(size)]
    times = []
    try:
        for name in TRANSPOSE_KERNELS:
            function = driver.function(module, name)
            driver.launch(function, grid, block, arguments)
            milliseconds = driver.time_launches(function, grid, block, arguments, repeats)
            times.append(TransposeTime(name, milliseconds, size))
    finally:
        driver.free(source)
        driver.free(target)
        driver.close()
    return times


@dataclasses.dataclass(frozen=True)
class MatmulTime:
    """The time of one matmul kernel on n by n matrices of halves."""

    kernel: str
    milliseconds: float
    size: int

    @property
    def teraflops(self) -> float:
        """The throughput the kernel reaches, counting a multiply and an add per term."""
        return 2 * self.size**3 / (self.milliseconds * 1e-3) / 1e12


def multiply(cubin_path: pathlib.Path, kernel: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a times the transpose of b in floats from one matmul kernel on the local GPU."""
    size = _matmul_size(a.shape[0])
    if a.shape != (size, size) or b.shape != (size, size):
        raise ValueError("the matmuls take two square matrices of one size")
    driver = Driver()
    try:
        module = driver.load(cubin_path.read_bytes())
        function, arguments, pointers = _matmul_launch(driver, module, kernel, a, b)
        try:
            driver.launch(function, *_matmul_grid(size), arguments)
            product = driver.download(pointers[0], 4 * size * size)
        finally:
            for pointer in pointers:
                driver.free(pointer)
    finally:
        driver.close()
    return np.frombuffer(product, dtype=np.float32).reshape(size, size)


def time_matmuls(cubin_path: pathlib.Path, size: int, repeats: int) -> list[MatmulTime]:
    """Time the two matmuls on random halves, after one launch of each to warm up.

    Raises RuntimeError when the two kernels give different products.
    """
    size = _matmul_size(size)
    rng = np.random.default_rng(0)
    a, b = (rng.uniform(-1, 1, (size, size)).astype(np.float16) for _ in range(2))
    driver = Driver()
    times, products = [], []
    try:
        module = driver.load(cubin_path.read_bytes())
        for name in MATMUL_KERNELS:
            function, arguments, pointers = _matmul_launch(driver, module, name, a, b)
            try:
                driver.launch(function, *_matmul_grid(size), arguments)
                products.append(driver.download(pointers[0], 4 * size * size))
                milliseconds = driver.time_launches(
                    function, *_matmul_grid(size), arguments, repeats
                )
            finally:
                for pointer in pointers:
                    driver.free(pointer)
            times.append(MatmulTime(name, milliseconds, size))
    finally:
        driver.close()
    if products[1] != products[0]:
        raise RuntimeError(f"{MATMUL_KERNELS[1]} gives another product than {MATMUL_KERNELS[0]}")
    return times


def _matmul_size(size: int) -> int:
    if size <= 0 or size % MATMUL_TILE:
        raise ValueError(f"the matrix size {size} must be a positive multiple of {MATMUL_TILE}")
    return size


def _matmul_grid(size: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    blocks = size // MATMUL_TILE
    return (blocks, blocks, 1), (MATMUL_THREADS, 1, 1)


def _matmul_launch(driver: Driver, module, kernel: str, a: np.ndarray, b: np.ndarray):
    size = a.shape[0]
    product = driver.alloc(4 * size * size)
    inputs = [driver.alloc(matrix.nbytes) for matrix in (a, b)]
    for pointer, matrix in zip(inputs, (a, b), strict=True):
        driver.upload(pointer, np.ascontiguousarray(matrix, dtype=np.float16).tobytes())
    pointers = [product, *inputs]
    arguments = [*(ctypes.c_uint64(pointer) for pointer in pointers), ctypes.c_int32(size)]
    return driver.function(module, kernel), arguments, pointers


def run(cubin_path: pathlib.Path, random_count: int = 0, seed: int = 0) -> None:
    """Run every case, the three transposes, and the two matmuls, one launch each, in order."""
    driver = Driver()
    module = driver.load(cubin_path.read_bytes())
    buffer = driver.alloc(GLOBAL_BYTES)
    driver.upload(buffer, bytes(GLOBAL_BYTES))
    for case in launches(random_count, seed):
        # The offsets travel by value in the parameter space, which keeps the kernel free of
        # memory instructions other than the one of the case.
        arguments = [
            (ctypes.c_int32 * WARP_SIZE)(*case.offsets),
            ctypes.c_uint32(case.mask),
            ctypes.c_uint32(SINK),
            ctypes.c_uint64(buffer),
        ]
        driver.launch(driver.function(module, case.kernel), (1, 1, 1), (WARP_SIZE, 1, 1), arguments)
    matrix_bytes = 4 * TRANSPOSE_SIZE * TRANSPOSE_SIZE
    source, target = driver.alloc(matrix_bytes), driver.alloc(matrix_bytes)
    tiles = TRANSPOSE_SIZE // 32
    for name in TRANSPOSE_KERNELS:
        arguments = [
            ctypes.c_uint64(target),
            ctypes.c_uint64(source),
            ctypes.c_int32(TRANSPOSE_SIZE),
        ]
        driver.launch(driver.function(module, name), (tiles, tiles, 1), (32, 8, 1), arguments)
    zeros = np.zeros((MATMUL_SIZE, MATMUL_SIZE), dtype=np.float16)
    for name in MATMUL_KERNELS:
        function, arguments, pointers = _matmul_launch(driver, module, name, zeros, zeros)
        driver.launch(function, *_matmul_grid(MATMUL_SIZE), arguments)
        for pointer in pointers:
            driver.free(pointer)
    for pointer in (buffer, source, target):
        driver.free(pointer)
    driver.close()


_PRELUDE = f"""
// Calibration kernels of LaneScope, generated by lanescope/nsight/calibration.py.
#define SHARED_BYTES {SHARED_BYTES}

struct LaneOffsets {{
    int value[{WARP_SIZE}];
}};
"""

_PTX_TYPE = {1: "u8", 2: "u16", 4: "u32", 8: "v2.u32", 16: "v4.u32"}
_REGISTERS = {1: 1, 2: 1, 4: 1, 8: 2, 16: 4}


def _access_kernel(space: Space, kind: Kind, width: int) -> str:
    count = _REGISTERS[width]
    names = ["a", "b", "c", "d"][:count]
    if space == Space.SHARED:
        address = (
            "const unsigned address = "
            "static_cast<unsigned>(__cvta_generic_to_shared(smem + offset));"
        )
        operand, state = "r", "volatile.shared"
    else:
        address = "unsigned char* address = buffer + offset;"
        operand, state = "l", "global"
    keep = ""
    if kind == Kind.LOAD:
        registers = "{" + ", ".join(f"%{i}" for i in range(count)) + "}" if count > 1 else "%0"
        asm = (
            f'asm volatile("ld.{state}.{_PTX_TYPE[width]} {registers}, [%{count}];" : '
            + ", ".join(f'"=r"({n})' for n in names)
            + f' : "{operand}"(address) : "memory");'
        )
        if space == Space.GLOBAL:
            keep = f'if (({" ^ ".join(names)}) == sink) asm volatile("trap;");'
    else:
        registers = "%1" if count == 1 else "{" + ", ".join(f"%{i + 1}" for i in range(count)) + "}"
        asm = (
            f'asm volatile("st.{state}.{_PTX_TYPE[width]} [%0], {registers};" :: '
            f'"{operand}"(address), ' + ", ".join(f'"r"({n})' for n in names) + ' : "memory");'
        )
    return f"""
extern "C" __global__ void {space.name.lower()}_{kind.name.lower()}_{width}(
    const __grid_constant__ LaneOffsets offsets, unsigned mask, unsigned sink,
    unsigned char* buffer) {{
    __shared__ __align__(16) unsigned char smem[SHARED_BYTES];
    const unsigned lane = threadIdx.x;
    unsigned a = lane, b = lane + 1, c = lane + 2, d = lane + 3;
    if ((mask >> lane) & 1u) {{
        const int offset = offsets.value[lane];
        {address}
        {asm}
        {keep}
    }}
}}
"""


def _ldmatrix_kernel(name: str) -> str:
    _, shape, *transpose = name.split("_")
    count = int(shape[1:])
    names = ["a", "b", "c", "d"][:count]
    qualifiers = ".".join(["m8n8", shape, *transpose])
    registers = ", ".join(f"%{i}" for i in range(count))
    outputs = ", ".join(f'"=r"({n})' for n in names)
    # ldmatrix needs every lane of the warp, so the kernel takes no mask.
    return f"""
extern "C" __global__ void {name}(
    const __grid_constant__ LaneOffsets offsets, unsigned mask, unsigned sink,
    unsigned char* buffer) {{
    __shared__ __align__(128) unsigned char smem[SHARED_BYTES];
    const unsigned address =
        static_cast<unsigned>(__cvta_generic_to_shared(smem + offsets.value[threadIdx.x]));
    unsigned {", ".join(names)};
    asm volatile("ldmatrix.sync.aligned.{qualifiers}.shared.b16 {{{registers}}}, [%{count}];"
                 : {outputs} : "r"(address) : "memory");
    if (({" ^ ".join(names)}) == sink) asm volatile("trap;");
}}
"""


def _cp_async_kernel(width: int) -> str:
    # cp.async.cg, which bypasses L1, copies 16 bytes only.
    cache = "cg" if width == 16 else "ca"
    return f"""
extern "C" __global__ void cp_async_{width}(
    const __grid_constant__ LaneOffsets offsets, unsigned mask, unsigned sink,
    unsigned char* buffer) {{
    __shared__ __align__(128) unsigned char smem[SHARED_BYTES];
    const unsigned address =
        static_cast<unsigned>(__cvta_generic_to_shared(smem + offsets.value[threadIdx.x]));
    asm volatile("cp.async.{cache}.shared.global [%0], [%1], {width};"
                 :: "r"(address), "l"(buffer + {width} * threadIdx.x) : "memory");
    asm volatile("cp.async.wait_all;" ::: "memory");
}}
"""


_TRANSPOSES = """
#define TILE 32
#define BLOCK_ROWS 8

extern "C" __global__ void transpose_tile(float* out, const float* in, int n) {
    __shared__ float tile[TILE][TILE];
    int x = blockIdx.x * TILE + threadIdx.x, y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
}

extern "C" __global__ void transpose_padded(float* out, const float* in, int n) {
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x, y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS)
        out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];
}

// Column c of row r lives at column c XOR r, the byte swizzle Swizzle<5,2,5> of LaneScope.
extern "C" __global__ void transpose_swizzled(float* out, const float* in, int n) {
    __shared__ float tile[TILE][TILE];
    int x = blockIdx.x * TILE + threadIdx.x, y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        const int row = threadIdx.y + j;
        tile[row][threadIdx.x ^ row] = in[(y + j) * n + x];
    }
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        const int column = threadIdx.y + j;
        out[(y + j) * n + x] = tile[threadIdx.x][column ^ threadIdx.x];
    }
}
"""


_MATMULS = """
// The kernels compute c = a * b^T for n x n matrices: a and b hold halves row by row, and
// c holds floats. Each block of 128 threads computes a 64 x 64 tile of c with mma.sync.
#define MM_TILE 64
#define MM_ROW_BYTES 128

// The byte offset of 16-byte chunk `chunk` of row `row` in a 64 x 64 tile of halves. The
// swizzled layout XORs the chunk with the low three bits of the row, Swizzle<3,4,3> in CuTe.
template <bool kSwizzled>
__device__ __forceinline__ unsigned chunk_offset(unsigned row, unsigned chunk) {
    return row * MM_ROW_BYTES + (kSwizzled ? chunk ^ (row & 7) : chunk) * 16;
}

template <bool kSwizzled>
__device__ __forceinline__ void matmul(float* c, const unsigned short* a, const unsigned short* b,
                                       int n) {
    __shared__ __align__(128) unsigned char tiles[2 * MM_TILE * MM_ROW_BYTES];
    const unsigned a_tile = static_cast<unsigned>(__cvta_generic_to_shared(tiles));
    const unsigned b_tile = a_tile + MM_TILE * MM_ROW_BYTES;
    const unsigned thread = threadIdx.x, lane = thread % 32, warp = thread / 32;
    const unsigned warp_row = warp / 2 * 32, warp_col = warp % 2 * 32;
    const size_t block_row = blockIdx.y * MM_TILE, block_col = blockIdx.x * MM_TILE;
    float acc[2][4][4] = {};
    for (int k0 = 0; k0 < n; k0 += MM_TILE) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned index = thread + 128 * i, row = index / 8, chunk = index % 8;
            const unsigned short* a_row = a + (block_row + row) * n + k0 + chunk * 8;
            const unsigned short* b_row = b + (block_col + row) * n + k0 + chunk * 8;
            const unsigned offset = chunk_offset<kSwizzled>(row, chunk);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                         :: "r"(a_tile + offset), "l"(a_row) : "memory");
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                         :: "r"(b_tile + offset), "l"(b_row) : "memory");
        }
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            unsigned fa[2][4], fb[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const unsigned row = warp_row + mi * 16 + lane % 16, chunk = kk * 2 + lane / 16;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                             : "=r"(fa[mi][0]), "=r"(fa[mi][1]), "=r"(fa[mi][2]), "=r"(fa[mi][3])
                             : "r"(a_tile + chunk_offset<kSwizzled>(row, chunk)) : "memory");
            }
#pragma unroll
            for (int nj = 0; nj < 2; ++nj) {
                const unsigned row = warp_col + nj * 16 + lane % 8 + lane / 16 * 8;
                const unsigned chunk = kk * 2 + lane / 8 % 2;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                             : "=r"(fb[nj][0]), "=r"(fb[nj][1]), "=r"(fb[nj][2]), "=r"(fb[nj][3])
                             : "r"(b_tile + chunk_offset<kSwizzled>(row, chunk)) : "memory");
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    float* d = acc[mi][ni];
                    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                        : "r"(fa[mi][0]), "r"(fa[mi][1]), "r"(fa[mi][2]), "r"(fa[mi][3]),
                          "r"(fb[ni / 2][ni % 2 * 2]), "r"(fb[ni / 2][ni % 2 * 2 + 1]));
                }
            }
        }
        __syncthreads();
    }
    const unsigned group = lane / 4, pair = lane % 4 * 2;
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const size_t row = block_row + warp_row + mi * 16 + group + half * 8;
                const size_t col = block_col + warp_col + ni * 8 + pair;
                *reinterpret_cast<float2*>(c + row * n + col) =
                    make_float2(acc[mi][ni][half * 2], acc[mi][ni][half * 2 + 1]);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(128)
matmul_plain(float* c, const unsigned short* a, const unsigned short* b, int n) {
    matmul<false>(c, a, b, n);
}

extern "C" __global__ void __launch_bounds__(128)
matmul_swizzled(float* c, const unsigned short* a, const unsigned short* b, int n) {
    matmul<true>(c, a, b, n);
}
"""
