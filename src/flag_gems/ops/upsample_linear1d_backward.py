import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scatter-style backward (mirrors the forward kernel).
#
# For each grad_output position the kernel computes the matching pair of
# grad_input positions and atomically adds the weighted contributions. This
# is optimal when out_w <= in_w (downsample-of-forward): only one output
# contributes per input, so atomic-adds never contend, and total atomic-add
# count is `out_w` per (n,c) — half of the gather kernel's traffic.
# ---------------------------------------------------------------------------
@triton.jit
def _scatter_backward_kernel(
    grad_out_ptr,
    grad_in_ptr,
    NC,
    W_in,
    W_out,
    scale,
    bias,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_w = tl.program_id(1)

    base_in = pid_nc * W_in
    base_out = pid_nc * W_out

    offs_w = pid_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (pid_nc < NC) & (offs_w < W_out)

    src = offs_w.to(tl.float32) * scale + bias
    src = tl.maximum(0.0, tl.minimum(src, W_in - 1.0))

    lower = tl.floor(src).to(tl.int32)
    upper = tl.minimum(lower + 1, W_in - 1)
    t = src - lower.to(tl.float32)
    w1 = t
    w0 = 1.0 - t

    g = tl.load(grad_out_ptr + base_out + offs_w, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_add(grad_in_ptr + base_in + lower, g * w0, mask=mask)
    tl.atomic_add(grad_in_ptr + base_in + upper, g * w1, mask=mask)


# ---------------------------------------------------------------------------
# Gather-style backward (each program owns a row tile, sums contributions
# from a 5-wide window of grad_output positions). Used when scale > 1 — the
# scatter path would do `2 * out_w = 4 * in_w` atomic-adds with heavy
# write-write contention on the same input slot, which is markedly slower
# on Ascend than the masked-load loop.
# ---------------------------------------------------------------------------
@triton.jit
def _gather_backward_kernel(
    grad_out_ptr,
    grad_in_ptr,
    nc,
    in_w,
    out_w,
    scale_fwd,
    scale_bwd,
    bias_bwd,
    align_corners: tl.constexpr,
    BLOCK_NC: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_w = tl.program_id(1)

    nc_offs = pid_nc * BLOCK_NC + tl.arange(0, BLOCK_NC)
    nc_mask = nc_offs < nc

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = offs_w < in_w

    mask2d = nc_mask[:, None] & w_mask[None, :]

    x_in_i = offs_w.to(tl.int32)
    x_in_f = offs_w.to(tl.float32)
    in_w_f = tl.cast(in_w, tl.float32)

    if align_corners:
        center = x_in_f * scale_fwd
    else:
        center = x_in_f * scale_fwd + (scale_fwd * 0.5 - 0.5)

    base = tl.floor(center).to(tl.int32)

    acc = tl.zeros([BLOCK_NC, BLOCK_W], dtype=tl.float32)

    for i in range(-2, 3):
        x_out = base + i
        valid = (x_out >= 0) & (x_out < out_w)
        x_out_f = x_out.to(tl.float32)

        if align_corners:
            x_real = x_out_f * scale_bwd
        else:
            x_real = x_out_f * scale_bwd + bias_bwd

        x0_f = tl.floor(x_real)
        w1 = x_real - x0_f
        w0 = 1.0 - w1

        x0_i = tl.maximum(x0_f, 0.0).to(tl.int32)
        x1_i = tl.minimum(x0_f + 1.0, in_w_f - 1.0).to(tl.int32)

        load_offs = nc_offs[:, None] * out_w + x_out[None, :]
        load_mask = mask2d & valid[None, :]
        g = tl.load(grad_out_ptr + load_offs, mask=load_mask, other=0.0).to(tl.float32)

        is_x0 = (x_in_i == x0_i)[None, :]
        is_x1 = (x_in_i == x1_i)[None, :]

        acc += tl.where(is_x0, g * w0[None, :], 0.0)
        acc += tl.where(is_x1, g * w1[None, :], 0.0)

    store_offs = nc_offs[:, None] * in_w + offs_w[None, :]
    tl.store(grad_in_ptr + store_offs, acc, mask=mask2d)


def _pick_gather_block_dims(nc: int, in_w: int, vendor_name: str):
    if vendor_name == "ascend":
        if in_w <= 128:
            block_w = triton.next_power_of_2(max(in_w, 32))
        elif in_w <= 2048:
            block_w = 512
        else:
            block_w = 1024
        block_nc = 2 if nc >= 2 else 1
        return block_nc, block_w
    return 1, 512


def _run_scatter(grad_out_2d, n, c, in_w, out_w, align_corners, vendor_name):
    nc = n * c
    # Atomic scatter needs a zero-initialised target; atomic_add on Ascend
    # is unreliable for bf16/fp16 pointers, so accumulate in fp32 and cast
    # back if the user asked for a narrower dtype.
    accumulate_in_fp32 = grad_out_2d.dtype != torch.float32
    grad_in = torch.zeros(
        (nc, in_w),
        device=grad_out_2d.device,
        dtype=torch.float32 if accumulate_in_fp32 else grad_out_2d.dtype,
    )

    if align_corners:
        scale = (in_w - 1.0) / (out_w - 1.0) if out_w > 1 else 0.0
        bias = 0.0
    else:
        scale = in_w / out_w
        bias = 0.5 * scale - 0.5

    block_size = 256
    grid = (nc, triton.cdiv(out_w, block_size))
    _scatter_backward_kernel[grid](
        grad_out_2d,
        grad_in,
        nc,
        in_w,
        out_w,
        scale,
        bias,
        BLOCK_SIZE=block_size,
    )
    if accumulate_in_fp32:
        grad_in = grad_in.to(grad_out_2d.dtype)
    return grad_in.view(n, c, in_w)


def _run_gather(grad_out_3d, n, c, in_w, out_w, align_corners, vendor_name):
    grad_in = torch.empty(
        (n, c, in_w), device=grad_out_3d.device, dtype=grad_out_3d.dtype
    )
    if align_corners:
        scale_fwd = (out_w - 1.0) / (in_w - 1.0) if in_w > 1 else 0.0
        scale_bwd = (in_w - 1.0) / (out_w - 1.0) if out_w > 1 else 0.0
        bias_bwd = 0.0
    else:
        scale_fwd = out_w / in_w
        scale_bwd = in_w / out_w
        bias_bwd = 0.5 * scale_bwd - 0.5

    nc = n * c
    block_nc, block_w = _pick_gather_block_dims(nc, in_w, vendor_name)
    grid = (triton.cdiv(nc, block_nc), triton.cdiv(in_w, block_w))
    _gather_backward_kernel[grid](
        grad_out_3d,
        grad_in,
        nc,
        in_w,
        out_w,
        scale_fwd,
        scale_bwd,
        bias_bwd,
        align_corners,
        BLOCK_NC=block_nc,
        BLOCK_W=block_w,
    )
    return grad_in


def upsample_linear1d_backward(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    align_corners: bool,
    scale_factors=None,
) -> torch.Tensor:
    logger.debug("GEMS UPSAMPLE_LINEAR1D_BACKWARD")

    if len(input_size) == 3:
        n, c, in_w = input_size
    elif len(input_size) == 2:
        n, c, in_w = input_size[0], 1, input_size[1]
    elif len(input_size) == 1:
        n, c, in_w = 1, 1, input_size[0]
    else:
        raise ValueError

    if output_size is not None:
        out_w = output_size[0]
    else:
        assert scale_factors is not None
        scale = (
            scale_factors[0]
            if isinstance(scale_factors, (list, tuple))
            else float(scale_factors)
        )
        out_w = int(in_w * scale)

    assert grad_output.shape[-1] == out_w

    vendor = runtime_device.vendor_name
    grad_out_contig = grad_output.contiguous()

    # Pick scatter when out_w <= in_w (downsample-of-forward direction):
    # each grad_input position has at most one contributor, so atomic-add
    # contention is zero and the kernel is much faster than gathering a
    # 5-wide window with mostly-wasted lanes. For up-sample direction
    # (out_w > in_w) the scatter path's contention dominates and gather
    # wins.
    if out_w <= in_w:
        return _run_scatter(
            grad_out_contig.view(n * c, out_w), n, c, in_w, out_w, align_corners, vendor
        )
    return _run_gather(
        grad_out_contig.view(n, c, out_w), n, c, in_w, out_w, align_corners, vendor
    )
