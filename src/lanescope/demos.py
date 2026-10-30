from collections.abc import Callable

from lanescope.trace import WARP_SIZE, Access, Kind, Space, Trace

TILE = 32
BLOCK_ROWS = 8
FLOAT_BYTES = 4
INPUT_BASE = 0x7F10_0000_0000
OUTPUT_BASE = 0x7F20_0000_0000

TRANSPOSE_SOURCE = (
    "__global__ void transpose(float *out, const float *in, int n) {",
    "  __shared__ float tile[32][32];",
    "  int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;",
    "  for (int j = 0; j < 32; j += 8)",
    "    tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * n + x];",
    "  __syncthreads();",
    "  x = blockIdx.y * 32 + threadIdx.x;",
    "  y = blockIdx.x * 32 + threadIdx.y;",
    "  for (int j = 0; j < 32; j += 8)",
    "    out[(y + j) * n + x] = tile[threadIdx.x][threadIdx.y + j];",
    "}",
)

# One PC per memory instruction of the kernel, with the source line it comes from.
PC_LOAD_INPUT = 0x0A0
PC_STORE_TILE = 0x0B0
PC_LOAD_TILE = 0x120
PC_STORE_OUTPUT = 0x130
TRANSPOSE_LINES = {PC_LOAD_INPUT: 5, PC_STORE_TILE: 5, PC_LOAD_TILE: 10, PC_STORE_OUTPUT: 10}


def warp_access(
    pc: int,
    space: Space,
    kind: Kind,
    width: int,
    address_of: Callable[[int], int | None],
    cta: int = 0,
    warp: int = 0,
) -> Access:
    """Build one warp-level access from a function of the lane index."""
    return Access(
        pc, space, kind, width, tuple(address_of(lane) for lane in range(WARP_SIZE)), cta, warp
    )


def transpose_trace(tiles_per_side: int = 2) -> Trace:
    """Return the trace of the tiled transpose on a square matrix of 32-by-32 tiles."""
    n = tiles_per_side * TILE
    accesses = []
    for block_y in range(tiles_per_side):
        for block_x in range(tiles_per_side):
            cta = block_y * tiles_per_side + block_x
            for warp in range(BLOCK_ROWS):
                accesses.extend(_transpose_warp(n, block_x, block_y, cta, warp))
    return _with_source(accesses, "transpose", TRANSPOSE_SOURCE, TRANSPOSE_LINES)


def _transpose_warp(n: int, block_x: int, block_y: int, cta: int, warp: int) -> list[Access]:
    # Warp w holds threadIdx.y == w, and its lane is threadIdx.x.
    accesses = []
    for j in range(0, TILE, BLOCK_ROWS):
        row = warp + j
        accesses.append(
            warp_access(
                PC_LOAD_INPUT,
                Space.GLOBAL,
                Kind.LOAD,
                FLOAT_BYTES,
                lambda lane, row=row: (
                    INPUT_BASE + FLOAT_BYTES * ((block_y * TILE + row) * n + block_x * TILE + lane)
                ),
                cta,
                warp,
            )
        )
        accesses.append(
            warp_access(
                PC_STORE_TILE,
                Space.SHARED,
                Kind.STORE,
                FLOAT_BYTES,
                lambda lane, row=row: FLOAT_BYTES * (row * TILE + lane),
                cta,
                warp,
            )
        )
    for j in range(0, TILE, BLOCK_ROWS):
        row = warp + j
        accesses.append(
            warp_access(
                PC_LOAD_TILE,
                Space.SHARED,
                Kind.LOAD,
                FLOAT_BYTES,
                lambda lane, row=row: FLOAT_BYTES * (lane * TILE + row),
                cta,
                warp,
            )
        )
        accesses.append(
            warp_access(
                PC_STORE_OUTPUT,
                Space.GLOBAL,
                Kind.STORE,
                FLOAT_BYTES,
                lambda lane, row=row: (
                    OUTPUT_BASE + FLOAT_BYTES * ((block_x * TILE + row) * n + block_y * TILE + lane)
                ),
                cta,
                warp,
            )
        )
    return accesses


ROW_SUMS_SOURCE = (
    "__global__ void row_sums(const float4 *in, float *out) {",
    "  __shared__ float4 tile[32][16];  // 32 rows of 64 floats",
    "  for (int k = threadIdx.x; k < 32 * 16; k += 32)",
    "    tile[k / 16][k % 16] = in[blockIdx.x * 512 + k];",
    "  __syncwarp();",
    "  float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);",
    "  for (int k = 0; k < 16; ++k) {",
    "    float4 v = tile[threadIdx.x][k];",
    "    sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;",
    "  }",
    "  out[blockIdx.x * 32 + threadIdx.x] = sum.x + sum.y + sum.z + sum.w;",
    "}",
)
ROW_SUMS_LINES = {0x070: 4, 0x080: 4, 0x110: 8, 0x1A0: 11}
ROW_BYTES = 16 * 16
VECTOR_BYTES = 16

COLUMN_SUMS_SOURCE = (
    "__global__ void column_sums(const float *in, float *out, int n) {",
    "  int column = blockIdx.x * 32 + threadIdx.x;",
    "  float sum = 0.f;",
    "  for (int row = 0; row < 32; ++row)",
    "    sum += in[column * n + row];",
    "  out[column] = sum;",
    "}",
)
COLUMN_SUMS_LINES = {0x0C0: 5, 0x150: 6}
COLUMN_SUMS_N = 1024


def row_sums_trace(blocks: int = 4) -> Trace:
    """Return the trace of a kernel whose lanes each read one 256-byte row of a shared tile."""
    load_in, store_tile, load_tile, store_out = sorted(ROW_SUMS_LINES)
    accesses = []
    for block in range(blocks):
        for step in range(VECTOR_BYTES):
            accesses.append(
                warp_access(
                    load_in,
                    Space.GLOBAL,
                    Kind.LOAD,
                    VECTOR_BYTES,
                    lambda lane, step=step, block=block: (
                        INPUT_BASE + VECTOR_BYTES * (block * 512 + step * WARP_SIZE + lane)
                    ),
                    block,
                )
            )
            accesses.append(
                warp_access(
                    store_tile,
                    Space.SHARED,
                    Kind.STORE,
                    VECTOR_BYTES,
                    lambda lane, step=step: VECTOR_BYTES * (step * WARP_SIZE + lane),
                    block,
                )
            )
        for column in range(16):
            accesses.append(
                warp_access(
                    load_tile,
                    Space.SHARED,
                    Kind.LOAD,
                    VECTOR_BYTES,
                    lambda lane, column=column: ROW_BYTES * lane + VECTOR_BYTES * column,
                    block,
                )
            )
        accesses.append(
            warp_access(
                store_out,
                Space.GLOBAL,
                Kind.STORE,
                FLOAT_BYTES,
                lambda lane, block=block: OUTPUT_BASE + FLOAT_BYTES * (block * WARP_SIZE + lane),
                block,
            )
        )
    return _with_source(accesses, "row_sums", ROW_SUMS_SOURCE, ROW_SUMS_LINES)


def column_sums_trace(blocks: int = 2) -> Trace:
    """Return the trace of a kernel whose lanes each read one column of a column-major matrix."""
    load_in, store_out = sorted(COLUMN_SUMS_LINES)
    accesses = []
    for block in range(blocks):
        for row in range(WARP_SIZE):
            accesses.append(
                warp_access(
                    load_in,
                    Space.GLOBAL,
                    Kind.LOAD,
                    FLOAT_BYTES,
                    lambda lane, row=row, block=block: (
                        INPUT_BASE
                        + FLOAT_BYTES * ((block * WARP_SIZE + lane) * COLUMN_SUMS_N + row)
                    ),
                    block,
                )
            )
        accesses.append(
            warp_access(
                store_out,
                Space.GLOBAL,
                Kind.STORE,
                FLOAT_BYTES,
                lambda lane, block=block: OUTPUT_BASE + FLOAT_BYTES * (block * WARP_SIZE + lane),
                block,
            )
        )
    return _with_source(accesses, "column_sums", COLUMN_SUMS_SOURCE, COLUMN_SUMS_LINES)


MATMUL_TILE_SOURCE = (
    "__global__ void matmul_tile(float *c, const half *a, int n) {",
    "  __shared__ half tile[64][64];  // 64 rows of 128 bytes",
    "  for (int i = 0; i < 4; ++i) {",
    "    int chunk = threadIdx.x + 128 * i;  // 8 chunks of 16 bytes per row",
    "    cp_async_16(&tile[chunk / 8][chunk % 8 * 8],",
    "                &a[(blockIdx.x * 64 + chunk / 8) * n + chunk % 8 * 8]);",
    "  }",
    "  cp_async_wait_all();",
    "  __syncthreads();",
    "  int lane = threadIdx.x % 32, row = threadIdx.x / 64 * 32;",
    "  for (int k = 0; k < 4; ++k)",
    "    ldmatrix_x4(fragment[k], &tile[row + lane % 16][k * 16 + lane / 16 * 8]);",
    "  // mma.sync multiplies the fragments, and the block stores its part of c",
    "}",
)
PC_COPY_TILE = 0x0A0
PC_READ_TILE = 0x1F0
MATMUL_TILE_LINES = {PC_COPY_TILE: 5, PC_READ_TILE: 12}
MATMUL_TILE_OPCODES = {PC_COPY_TILE: "LDGSTS.E.BYPASS.128", PC_READ_TILE: "LDSM.16.M88.4"}
TILE_ROW_BYTES = 128
CHUNK_BYTES = 16


def matmul_tile_trace(blocks: int = 2) -> Trace:
    """Return the trace of a matrix multiply that copies a 64 x 64 tile and reads it with ldmatrix.

    The tile keeps the plain row-major layout, so the eight rows of each 8x8 matrix that
    ldmatrix reads fall on the same banks.
    """
    accesses = []
    for block in range(blocks):
        for warp in range(4):
            for step in range(4):

                def chunk(lane: int, warp: int = warp, step: int = step) -> int:
                    return WARP_SIZE * warp + lane + 128 * step

                def source(lane: int, block: int = block) -> int:
                    row, column = divmod(chunk(lane), 8)
                    return INPUT_BASE + (64 * block + row) * TILE_ROW_BYTES + column * CHUNK_BYTES

                accesses.append(
                    warp_access(PC_COPY_TILE, Space.GLOBAL, Kind.LOAD, 16, source, block, warp)
                )
                accesses.append(
                    warp_access(
                        PC_COPY_TILE,
                        Space.SHARED,
                        Kind.STORE,
                        16,
                        lambda lane: 16 * chunk(lane),
                        block,
                        warp,
                    )
                )
            for k in range(4):

                def row_of(lane: int, warp: int = warp, k: int = k) -> int:
                    row = warp // 2 * 32 + lane % 16
                    return row * TILE_ROW_BYTES + (2 * k + lane // 16) * CHUNK_BYTES

                accesses.append(
                    warp_access(PC_READ_TILE, Space.SHARED, Kind.LOAD, 16, row_of, block, warp)
                )
    return _with_source(
        accesses, "matmul_tile", MATMUL_TILE_SOURCE, MATMUL_TILE_LINES, MATMUL_TILE_OPCODES
    )


def demo_traces() -> dict[str, Trace]:
    """Return every demo trace by the file name it takes.

    The numbers in the names put the transpose first in a listing, the one demo with
    the textbook bank conflict and both kinds of fix.
    """
    return {
        "1_transpose.npz": transpose_trace(),
        "2_row_sums.npz": row_sums_trace(),
        "3_column_sums.npz": column_sums_trace(),
        "4_matmul_tile.npz": matmul_tile_trace(),
    }


def _with_source(
    accesses: list[Access],
    kernel: str,
    source: tuple[str, ...],
    lines: dict[int, int],
    opcodes: dict[int, str] | None = None,
) -> Trace:
    metadata: dict = {
        "kernel": kernel,
        "arch": "synthetic",
        "source": list(source),
        "lines": {f"{pc:#x}": line for pc, line in lines.items()},
    }
    if opcodes:
        metadata["opcodes"] = {f"{pc:#x}": opcode for pc, opcode in opcodes.items()}
    return Trace.from_accesses(accesses, metadata)
