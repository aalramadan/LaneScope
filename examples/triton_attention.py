import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(q, k, v, o, length, scale, BLOCK: tl.constexpr, HEAD: tl.constexpr):
    """Compute BLOCK rows of softmax(q k^T * scale) v, one block of keys at a time."""
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    dims = tl.arange(0, HEAD)
    query = tl.load(q + rows[:, None] * HEAD + dims[None, :])
    best = tl.full((BLOCK,), -float("inf"), tl.float32)
    total = tl.zeros((BLOCK,), tl.float32)
    output = tl.zeros((BLOCK, HEAD), tl.float32)
    for start in range(0, length, BLOCK):
        keys = start + tl.arange(0, BLOCK)
        key = tl.load(k + keys[None, :] * HEAD + dims[:, None])
        scores = tl.dot(query, key) * scale
        newest = tl.maximum(best, tl.max(scores, axis=1))
        weights = tl.exp(scores - newest[:, None])
        shrink = tl.exp(best - newest)
        total = total * shrink + tl.sum(weights, axis=1)
        value = tl.load(v + keys[:, None] * HEAD + dims[None, :])
        output = output * shrink[:, None] + tl.dot(weights.to(tl.float16), value)
        best = newest
    tl.store(o + rows[:, None] * HEAD + dims[None, :], output / total[:, None])


def main() -> None:
    """Run attention over one head of random queries, keys, and values, checked against PyTorch."""
    parser = argparse.ArgumentParser(description="attention over one head")
    parser.add_argument("--length", type=int, default=256, help="tokens, a multiple of 64")
    args = parser.parse_args()
    q, k, v = (torch.randn(args.length, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    o = torch.empty(args.length, 64, device="cuda", dtype=torch.float32)
    scale = 64**-0.5
    attention_kernel[(args.length // 64,)](q, k, v, o, args.length, scale, BLOCK=64, HEAD=64)
    expected = torch.softmax(q.float() @ k.float().T * scale, dim=-1) @ v.float()
    torch.testing.assert_close(o, expected, rtol=2e-2, atol=2e-2)
    print(f"attention over {args.length} tokens matches PyTorch")


if __name__ == "__main__":
    main()
