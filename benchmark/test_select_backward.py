import math

import torch
import triton
import triton.testing

import flag_gems
from flag_gems.ops.select_backward import select_backward

SHAPES = [
    (128, 256),
    (64, 128, 256),
    (1024, 4096),
    (32, 64, 128, 256),
]

DTYPES = [
    torch.float32,
    torch.float16,
]

DIMS = [0, 1, -1]


SIZES = [math.prod(s) for s in SHAPES]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=SIZES,
        line_arg="provider",
        line_vals=["pytorch", "triton"],
        line_names=["PyTorch", "FlagGems"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="Bandwidth (GB/s)",
        plot_name="select_backward_performance",
        args={},
    )
)
def benchmark_select_backward(size, provider, device=flag_gems.device):

    shape = None
    for s in SHAPES:
        if math.prod(s) == size:
            shape = s
            break

    if shape is None:
        raise RuntimeError("shape not found")

    dtype = torch.float16
    dim = 1

    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)

    ndim = len(shape)
    actual_dim = dim if dim >= 0 else dim + ndim

    index = shape[actual_dim] // 2

    y = torch.select(x, actual_dim, index)
    grad = torch.randn_like(y)

    out = torch.empty_like(x)

    element_size = grad.element_size()

    bytes_moved = (grad.numel() + x.numel()) * element_size

    quantiles = [0.5, 0.2, 0.8]

    if provider == "pytorch":

        def torch_impl():

            if x.grad is not None:
                x.grad.zero_()

            y.backward(grad, retain_graph=True)

        ms, min_ms, max_ms = triton.testing.do_bench(
            torch_impl,
            rep=100,
            quantiles=quantiles,
        )

    elif provider == "triton":

        def triton_impl():

            select_backward(
                grad,
                x.shape,
                actual_dim,
                index,
                out=out,
            )

        ms, min_ms, max_ms = triton.testing.do_bench(
            triton_impl,
            rep=100,
            quantiles=quantiles,
        )

    def gbps(ms):
        return bytes_moved / (ms * 1e-3) / 1e9

    return gbps(ms), gbps(min_ms), gbps(max_ms)


if __name__ == "__main__":

    benchmark_select_backward.run(
        print_data=True,
        save_path="./benchmark_results",
    )
