import argparse

import torch
import torch.nn.functional as F

OPS = ("matmul", "softmax", "layer_norm", "conv2d")


def run(op: str) -> tuple[int, ...]:
    """Run one PyTorch operation on small half-precision tensors and return its output shape."""
    torch.manual_seed(0)
    if op == "matmul":
        a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        b = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        out = a @ b
    elif op == "softmax":
        out = torch.softmax(torch.randn(256, 1024, device="cuda", dtype=torch.float16), dim=-1)
    elif op == "layer_norm":
        out = F.layer_norm(torch.randn(256, 1024, device="cuda", dtype=torch.float16), (1024,))
    else:
        images = torch.randn(8, 64, 32, 32, device="cuda", dtype=torch.float16)
        weights = torch.randn(64, 64, 3, 3, device="cuda", dtype=torch.float16)
        out = F.conv2d(images, weights, padding=1)
    torch.cuda.synchronize()
    return tuple(out.shape)


def main() -> None:
    """Run the operation named on the command line."""
    parser = argparse.ArgumentParser(description="run one PyTorch operation on the GPU")
    parser.add_argument("op", choices=OPS)
    op = parser.parse_args().op
    print(f"{op} gives a tensor of shape {run(op)}")


if __name__ == "__main__":
    main()
