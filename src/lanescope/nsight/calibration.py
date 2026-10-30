import ctypes
import dataclasses
import pathlib
import random

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

    @property
    def kernel(self) -> str:
        """Return the name of the kernel that runs this case."""
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


def launches(random_count: int = 0, seed: int = 0) -> list[Case]:
    """Return the cases of a calibration run, the fixed ones first and then the random ones."""
    return cases() + random_cases(random_count, seed)


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
    return "\n".join([_PRELUDE, *kernels, _TRANSPOSES])


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


def run(cubin_path: pathlib.Path, random_count: int = 0, seed: int = 0) -> None:
    """Run every case and then the three transposes, one launch each, in a fixed order."""
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
