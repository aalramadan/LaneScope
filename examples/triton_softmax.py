import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x, y, columns, BLOCK: tl.constexpr):
    """Write the softmax of one row of x to the same row of y."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    inside = offsets < columns
    values = tl.load(x + row * columns + offsets, mask=inside, other=-float("inf"))
    values = tl.exp(values - tl.max(values, axis=0))
    tl.store(y + row * columns + offsets, values / tl.sum(values, axis=0), mask=inside)


def main() -> None:
    """Take the softmax of each row of a random matrix and check it against PyTorch."""
    parser = argparse.ArgumentParser(description="softmax of each row of a matrix")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--columns", type=int, default=1000)
    args = parser.parse_args()
    x = torch.randn(args.rows, args.columns, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    block = triton.next_power_of_2(args.columns)
    softmax_kernel[(args.rows,)](x, y, args.columns, BLOCK=block, num_warps=4)
    torch.testing.assert_close(y, torch.softmax(x, dim=-1))
    print(f"the softmax of {args.rows} rows matches PyTorch")


if __name__ == "__main__":
    main()
