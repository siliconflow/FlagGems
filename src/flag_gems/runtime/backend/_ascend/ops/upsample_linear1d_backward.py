import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.runtime.backend._ascend.utils import CORE_NUM
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(
    f'flag_gems.runtime.backend._ascend.ops.{__name__.split(".")[-1]}'
)
device = device.name

# Tiling budget for the generic elementwise gather kernel.
ASCEND_910B_UB_BYTES = 192 * 1024
UB_RESERVED_BYTES = 16 * 1024
TEMP_BYTES_PER_LANE = 288
MAX_ROWS_PER_BLOCK = 16
MIN_BLOCK_W = 64

# GEMM (cube) path: only taken for exact x2 up/down-sampling, where the
# interpolation weights are dyadic and therefore stored bit-exactly in the
# grad_output dtype. The coeff-element cap bounds the materialized [out_w, in_w]
# matrix; beyond it the generic gather kernel is used instead.
DOT_COEFF_MAX_ELEMENTS = 64 * 1024 * 1024
DOT_MIN_ROWS = 16
DOT_BLOCK_M = 128
DOT_BLOCK_N = 256
DOT_BLOCK_K = 64


@triton.jit
def upsample_linear1d_backward_coeff_kernel(
    coeff_ptr,
    IN_W: tl.constexpr,
    OUT_W: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < (OUT_W * IN_W)
    x_out = offsets // IN_W
    x_in = offsets - x_out * IN_W

    x_out_f = x_out.to(tl.float32)
    if ALIGN_CORNERS:
        if OUT_W > 1:
            x_real = x_out_f * ((IN_W - 1) + 0.0) / ((OUT_W - 1) + 0.0)
        else:
            x_real = tl.zeros((BLOCK,), dtype=tl.float32)
    else:
        x_real = (x_out_f + 0.5) * (IN_W + 0.0) / (OUT_W + 0.0) - 0.5
        x_real = tl.maximum(x_real, 0.0)

    x0_f = tl.floor(x_real)
    w1 = x_real - x0_f
    w0 = 1.0 - w1
    x0 = tl.maximum(x0_f, 0.0).to(tl.int32)
    x1 = tl.minimum(x0_f + 1.0, (IN_W - 1) + 0.0).to(tl.int32)
    same = x0 == x1

    weight = tl.where(same & (x_in == x0), w0 + w1, 0.0)
    weight += tl.where((~same) & (x_in == x0), w0, 0.0)
    weight += tl.where((~same) & (x_in == x1), w1, 0.0)
    tl.store(coeff_ptr + offsets, weight, mask=mask)


@triton.jit
def upsample_linear1d_backward_dot_kernel(
    grad_out_ptr,
    coeff_ptr,
    grad_in_ptr,
    rows,
    IN_W: tl.constexpr,
    OUT_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    x_in_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, OUT_W, BLOCK_K):
        x_out_offsets = k_start + tl.arange(0, BLOCK_K)
        grad = tl.load(
            grad_out_ptr + rows_offsets[:, None] * OUT_W + x_out_offsets[None, :],
            mask=(rows_offsets[:, None] < rows) & (x_out_offsets[None, :] < OUT_W),
            other=0.0,
        )
        coeff = tl.load(
            coeff_ptr + x_out_offsets[:, None] * IN_W + x_in_offsets[None, :],
            mask=(x_out_offsets[:, None] < OUT_W) & (x_in_offsets[None, :] < IN_W),
            other=0.0,
        )
        acc += tl.dot(grad, coeff)

    tl.store(
        grad_in_ptr + rows_offsets[:, None] * IN_W + x_in_offsets[None, :],
        acc,
        mask=(rows_offsets[:, None] < rows) & (x_in_offsets[None, :] < IN_W),
    )


@triton.jit
def upsample_linear1d_backward_kernel(
    grad_out_ptr,
    grad_in_ptr,
    rows,
    in_w,
    out_w,
    align_corners: tl.constexpr,
    BLOCK_W: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
):
    row_start = ext.program_id(axis=1) * ROWS_PER_BLOCK
    row_step = tl.num_programs(axis=1) * ROWS_PER_BLOCK
    x_in = ext.program_id(axis=0) * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]
    width_mask = x_in < in_w

    x_in_f = x_in.to(tl.float32)
    in_w_f = tl.cast(in_w, tl.float32)
    out_w_f = tl.cast(out_w, tl.float32)

    if align_corners:
        if in_w > 1:
            center = x_in_f * (out_w_f - 1.0) / (in_w_f - 1.0)
        else:
            center = tl.zeros((1, BLOCK_W), dtype=tl.float32)
    else:
        center = (x_in_f + 0.5) * out_w_f / in_w_f - 0.5

    base = tl.floor(center).to(tl.int32)

    while row_start < rows:
        row_offsets = row_start + tl.arange(0, ROWS_PER_BLOCK)[:, None]
        mask = (row_offsets < rows) & width_mask
        go_base = grad_out_ptr + row_offsets * out_w
        acc = tl.zeros((ROWS_PER_BLOCK, BLOCK_W), dtype=tl.float32)

        for i in range(-2, 3):
            x_out = base + i
            valid = (x_out >= 0) & (x_out < out_w)
            x_out_f = x_out.to(tl.float32)

            if align_corners:
                if out_w > 1:
                    x_real = x_out_f * (in_w_f - 1.0) / (out_w_f - 1.0)
                else:
                    x_real = tl.zeros((1, BLOCK_W), dtype=tl.float32)
            else:
                x_real = (x_out_f + 0.5) * in_w_f / out_w_f - 0.5

            x0_f = tl.floor(x_real)
            w1 = x_real - x0_f
            w0 = 1.0 - w1

            x0_i = tl.maximum(x0_f, 0.0).to(tl.int32)
            x1_i = tl.minimum(x0_f + 1.0, in_w_f - 1.0).to(tl.int32)

            g = tl.load(go_base + x_out, mask=mask & valid, other=0.0).to(tl.float32)

            same = x0_i == x1_i
            is_x0 = x_in.to(tl.int32) == x0_i
            is_x1 = x_in.to(tl.int32) == x1_i

            acc += tl.where(same & is_x0, g * (w0 + w1), 0.0)
            acc += tl.where(~same & is_x0, g * w0, 0.0)
            acc += tl.where(~same & is_x1, g * w1, 0.0)

        tl.store(grad_in_ptr + row_offsets * in_w + x_in, acc, mask=mask)
        row_start += row_step


def _normalize_input_size(input_size):
    if len(input_size) == 3:
        return input_size
    if len(input_size) == 2:
        return input_size[0], 1, input_size[1]
    if len(input_size) == 1:
        return 1, 1, input_size[0]
    raise ValueError


def _device_guard(tensor):
    device_index = tensor.device.index
    if device_index is None or device_index == torch_device_fn.current_device():
        return nullcontext()
    return torch_device_fn.device(tensor.device)


def _prev_power_of_2(value):
    return 1 << (int(value).bit_length() - 1)


def _select_tiles(in_w, element_size):
    bytes_per_lane = TEMP_BYTES_PER_LANE + element_size * 8
    usable_ub = ASCEND_910B_UB_BYTES - UB_RESERVED_BYTES
    max_tile_elements = _prev_power_of_2(max(1, usable_ub // bytes_per_lane))
    block_w = min(max(MIN_BLOCK_W, triton.next_power_of_2(in_w)), max_tile_elements)
    rows_per_block = max(1, min(MAX_ROWS_PER_BLOCK, max_tile_elements // block_w))
    return block_w, rows_per_block


def _can_use_dot_path(grad_output, rows, in_w, out_w):
    if grad_output.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if not (out_w * 2 == in_w or out_w == in_w * 2):
        return False
    if rows < DOT_MIN_ROWS:
        return False
    return in_w * out_w <= DOT_COEFF_MAX_ELEMENTS


def _dot_block_k(grad_output, in_w, out_w):
    if (
        out_w * 2 == in_w
        and in_w >= 4096
        and grad_output.dtype
        in (
            torch.float16,
            torch.bfloat16,
        )
    ):
        return 128
    return DOT_BLOCK_K


def _upsample_linear1d_backward_dot(
    grad_out_3d, n, c, rows, in_w, out_w, align_corners
):
    grad_out_2d = grad_out_3d.view(rows, out_w)
    grad_in_2d = torch.empty(
        (rows, in_w), device=grad_out_3d.device, dtype=grad_out_3d.dtype
    )
    coeff = torch.empty(
        (out_w, in_w), device=grad_out_3d.device, dtype=grad_out_3d.dtype
    )

    upsample_linear1d_backward_coeff_kernel[(triton.cdiv(out_w * in_w, 1024),)](
        coeff,
        IN_W=in_w,
        OUT_W=out_w,
        ALIGN_CORNERS=align_corners,
        BLOCK=1024,
    )

    grid = (
        triton.cdiv(rows, DOT_BLOCK_M),
        triton.cdiv(in_w, DOT_BLOCK_N),
    )
    upsample_linear1d_backward_dot_kernel[grid](
        grad_out_2d,
        coeff,
        grad_in_2d,
        rows,
        IN_W=in_w,
        OUT_W=out_w,
        BLOCK_M=DOT_BLOCK_M,
        BLOCK_N=DOT_BLOCK_N,
        BLOCK_K=_dot_block_k(grad_out_3d, in_w, out_w),
    )
    return grad_in_2d.view(n, c, in_w)


def upsample_linear1d_backward(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    align_corners: bool,
    scale_factors=None,
) -> torch.Tensor:
    logger.debug("GEMS_ASCEND UPSAMPLE_LINEAR1D_BACKWARD")
    assert grad_output.device.type == device

    n, c, in_w = _normalize_input_size(input_size)
    if output_size is not None:
        out_w = output_size[0]
    else:
        assert scale_factors is not None
        out_w = int(in_w * scale_factors[0])

    assert grad_output.shape[-1] == out_w

    grad_out_3d = grad_output.contiguous().view(n, c, out_w)
    rows = n * c

    with _device_guard(grad_output):
        if _can_use_dot_path(grad_output, rows, in_w, out_w):
            # Exact-x2 shapes whose coeff matrix fits the budget: build the
            # [out_w, in_w] interpolation matrix and run grad_in = grad_out @ coeff
            # on the cube engine.
            grad_in = _upsample_linear1d_backward_dot(
                grad_out_3d, n, c, rows, in_w, out_w, align_corners
            )
        else:
            # Generic per-element gather: correct for any scale / align mode.
            grad_in = torch.empty(
                (n, c, in_w), device=grad_output.device, dtype=grad_output.dtype
            )
            block_w, rows_per_block = _select_tiles(in_w, grad_output.element_size())
            row_blocks = triton.cdiv(rows, rows_per_block)
            grid = (triton.cdiv(in_w, block_w), min(row_blocks, CORE_NUM))

            upsample_linear1d_backward_kernel[grid](
                grad_out_3d,
                grad_in,
                rows,
                in_w,
                out_w,
                align_corners,
                BLOCK_W=block_w,
                ROWS_PER_BLOCK=rows_per_block,
            )

    return grad_in
