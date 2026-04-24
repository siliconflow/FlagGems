import math

import torch
import triton
import triton.language as tl


@triton.jit
def _select_backward_kernel(
    grad_ptr,
    out_ptr,
    total: tl.constexpr,
    inner_size: tl.constexpr,
    dim_stride: tl.constexpr,
    index: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    outer = offs // inner_size
    inner = offs % inner_size

    vals = tl.load(grad_ptr + offs, mask=mask)
    out_offset = outer * dim_stride + index * inner_size + inner

    tl.store(out_ptr + out_offset, vals, mask=mask)


@triton.jit
def _select_backward_dim0_kernel(
    grad_ptr,
    out_ptr,
    total: tl.constexpr,
    base: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    vals = tl.load(grad_ptr + offs, mask=mask)
    tl.store(out_ptr + base + offs, vals, mask=mask)


@triton.jit
def _select_backward_lastdim_kernel(
    grad_ptr,
    out_ptr,
    total: tl.constexpr,
    dim_size: tl.constexpr,
    index: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    vals = tl.load(grad_ptr + offs, mask=mask)
    out_offset = offs * dim_size + index

    tl.store(out_ptr + out_offset, vals, mask=mask)


@triton.jit
def _select_backward_mid_large_inner_kernel(
    grad_ptr,
    out_ptr,
    inner_size: tl.constexpr,
    dim_size: tl.constexpr,
    index: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    pid_outer = tl.program_id(0)
    pid_inner = tl.program_id(1)

    inner = pid_inner * BLOCK_INNER + tl.arange(0, BLOCK_INNER)
    mask = inner < inner_size

    grad_offset = pid_outer * inner_size + inner
    out_offset = (pid_outer * dim_size + index) * inner_size + inner

    vals = tl.load(grad_ptr + grad_offset, mask=mask)
    tl.store(out_ptr + out_offset, vals, mask=mask)


def _launch_select_backward(grad, input_sizes, dim, index, out=None):
    dim = int(dim)
    index = int(index)

    sizes = list(input_sizes)
    ndim = len(sizes)

    if dim < 0:
        dim += ndim

    if dim < 0 or dim >= ndim:
        raise ValueError("invalid dim")

    dim_size = sizes[dim]

    if index < 0:
        index += dim_size

    if index < 0 or index >= dim_size:
        raise ValueError("index out of range")

    outer_size = math.prod(sizes[:dim]) if dim > 0 else 1
    inner_size = math.prod(sizes[dim + 1:]) if dim < ndim - 1 else 1
    total = outer_size * inner_size
    out_numel = math.prod(sizes)

    if not grad.is_contiguous():
        grad = grad.contiguous()

    grad_view = grad.view(total)

    if out is None:
        out = torch.empty(sizes, dtype=grad.dtype, device=grad.device)
    else:
        if tuple(out.shape) != tuple(sizes):
            raise ValueError("out shape mismatch")
        if out.dtype != grad.dtype:
            raise ValueError("dtype mismatch")
        if out.device != grad.device:
            raise ValueError("device mismatch")

    # 小shape：避免Triton自定义kernel路径
    if out_numel <= 4096:
        out.zero_()
        out.select(dim, index).copy_(grad)
        return out

    # 大shape：zero + copy-back
    out.zero_()

    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)

    if dim == 0:
        base = index * inner_size
        _select_backward_dim0_kernel[grid](
            grad_view,
            out,
            total,
            base,
            BLOCK=BLOCK,
        )
    elif inner_size == 1:
        _select_backward_lastdim_kernel[grid](
            grad_view,
            out,
            total,
            dim_size,
            index,
            BLOCK=BLOCK,
        )
    elif (
        inner_size >= 4096
        and outer_size <= 128
        and dim_size <= 128
    ):
        BLOCK_INNER = 1024
        grid = (outer_size, triton.cdiv(inner_size, BLOCK_INNER))

        _select_backward_mid_large_inner_kernel[grid](
            grad_view,
            out,
            inner_size,
            dim_size,
            index,
            BLOCK_INNER=BLOCK_INNER,
        )
    else:
        dim_stride = dim_size * inner_size
        _select_backward_kernel[grid](
            grad_view,
            out,
            total,
            inner_size,
            dim_stride,
            index,
            BLOCK=BLOCK,
        )

    return out


def select_backward(grad, input_sizes, dim, index, out=None):
    return _launch_select_backward(grad, input_sizes, dim, index, out=out)
