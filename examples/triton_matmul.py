import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a, b, c, n, BLOCK: tl.constexpr, BLOCK_K: tl.constexpr):
    """Compute one BLOCK x BLOCK tile of c = a @ b, taking BLOCK_K columns of a per step."""
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    steps = tl.arange(0, BLOCK_K)
    total = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for k in range(0, n, BLOCK_K):
        x = tl.load(a + rows[:, None] * n + (k + steps)[None, :])
        y = tl.load(b + (k + steps)[:, None] * n + cols[None, :])
        total += tl.dot(x, y)
    tl.store(c + rows[:, None] * n + cols[None, :], total)


def main() -> None:
    """Multiply two random matrices on the GPU and check the product against PyTorch."""
    parser = argparse.ArgumentParser(description="multiply two square matrices of halves")
    parser.add_argument("--size", type=int, default=256, help="rows and columns, a multiple of 64")
    parser.add_argument("--block-k", type=int, default=32, help="columns of a read per step")
    args = parser.parse_args()
    n = args.size
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)
    c = torch.empty(n, n, device="cuda", dtype=torch.float32)
    grid = (n // 64, n // 64)
    matmul_kernel[grid](a, b, c, n, BLOCK=64, BLOCK_K=args.block_k, num_warps=4, num_stages=3)
    torch.testing.assert_close(c, a.float() @ b.float(), rtol=1e-2, atol=1e-2)
    print(f"the {n} x {n} product matches PyTorch")


if __name__ == "__main__":
    main()
