import math

import torch
import triton
import triton.language as tl

@triton.jit
def _cuda_select_backward_kernel(
    grad_ptr,
    out_ptr,
    outer_size,
    inner_size,
    dim_stride,
    index,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = outer_size * inner_size

    mask = offs < total

    outer = offs // inner_size
    inner = offs % inner_size

    grad_vals = tl.load(grad_ptr + outer * inner_size + inner, mask=mask)

    out_offset = outer * dim_stride + index * inner_size + inner

    tl.store(out_ptr + out_offset, grad_vals, mask=mask)


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


@triton.jit
def _select_backward_fused_kernel(
    grad_ptr,
    out_ptr,
    out_numel: tl.constexpr,
    inner_size: tl.constexpr,
    dim_size: tl.constexpr,
    index: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < out_numel

    inner = offs % inner_size
    tmp = offs // inner_size
    dim_idx = tmp % dim_size
    outer = tmp // dim_size

    selected = dim_idx == index
    grad_offset = outer * inner_size + inner

    vals = tl.load(
        grad_ptr + grad_offset,
        mask=mask & selected,
        other=0.0,
    )
    vals = tl.where(selected, vals, 0.0)

    tl.store(out_ptr + offs, vals, mask=mask)


def _is_cuda_device(grad):
    return grad.device.type == "cuda"


def _is_ascend_device(grad):
    # torch_npu一般是npu；如果你的环境里device.type不是npu，可以在这里补充。
    return grad.device.type == "npu"


def _launch_select_backward(grad, input_sizes, dim, index, out=None):
    dim = int(dim)
    index = int(index)

    sizes = list(input_sizes)
    ndim = len(sizes)

    if dim < 0:
        dim += ndim

    if dim < 0 or dim >= ndim:
        raise ValueError("invalid dim")

    dim_size = int(sizes[dim])

    if index < 0:
        index += dim_size

    if index < 0 or index >= dim_size:
        raise ValueError("index out of range")

    outer_size = math.prod(sizes[:dim]) if dim > 0 else 1
    inner_size = math.prod(sizes[dim + 1:]) if dim < ndim - 1 else 1
    out_numel = math.prod(sizes)

    outer_size = int(outer_size)
    inner_size = int(inner_size)
    out_numel = int(out_numel)

    total = outer_size * inner_size

    if not grad.is_contiguous():
        grad = grad.contiguous()

    grad_view = grad.view(total)

    if out is None:
        out = torch.empty(
            sizes,
            dtype=grad.dtype,
            device=grad.device,
        )
    else:
        if tuple(out.shape) != tuple(sizes):
            raise ValueError("out shape mismatch")
        if out.dtype != grad.dtype:
            raise ValueError("dtype mismatch")
        if out.device != grad.device:
            raise ValueError("device mismatch")

    is_cuda = _is_cuda_device(grad)
    is_ascend = _is_ascend_device(grad)

    # 非CUDA小shape：保守走zero_+select.copy_。
    # Ascend上fused小shape之前容易拖慢精度测试和性能测试，所以不默认启用。
    if out_numel <= 4096:
        out.zero_()
        out.select(dim, index).copy_(grad)
        return out

    # 大shape：先全量清零，再只写回被select选中的切片。
    out.zero_()

    if total == 0:
        return out

    # dim=0：写回区域是完整连续块。
    if dim == 0:
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)
        base = index * inner_size

        _select_backward_dim0_kernel[grid](
            grad_view,
            out,
            total,
            base,
            BLOCK=BLOCK,
        )
        return out

    # inner_size=1：最后一维select，专门kernel避免通用kernel里的除法和取模。
    if inner_size == 1:
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)

        _select_backward_lastdim_kernel[grid](
            grad_view,
            out,
            total,
            dim_size,
            index,
            BLOCK=BLOCK,
        )
        return out

    # Ascend专用：修复outer较小、inner很大的中间维度大块写回。
    # 典型case：
    # grad=[64,4096], input=[64,64,4096], dim=1, index=32
    # CUDA上通用kernel已经很好，所以这里不要给CUDA启用，避免反向拖慢。
    if (
        is_ascend
        and inner_size >= 4096
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
        return out

    # 默认通用路径。
    # CUDA上这个路径对inner_size=16/256/512/1024/4096都表现很好；
    # Ascend上除了inner_size特别大的case，其余也相对稳定。
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
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


def _cuda_launch_select_backward(grad, input_sizes, dim, index, out=None):
    if not grad.is_cuda:
        raise ValueError("grad must be CUDA tensor")

    dim = int(dim)
    index = int(index)

    sizes = list(input_sizes)
    ndim = len(sizes)

    if dim < 0:
        dim += ndim

    if dim < 0 or dim >= ndim:
        raise ValueError("invalid dim")

    dim_size = sizes[dim]

    if index < 0 or index >= dim_size:
        raise ValueError("index out of range")

    outer_size = math.prod(sizes[:dim]) if dim > 0 else 1
    inner_size = math.prod(sizes[dim + 1 :]) if dim < ndim - 1 else 1

    grad_view = grad.contiguous().view(outer_size, inner_size)

    if out is None:
        out = torch.zeros(
            sizes,
            dtype=grad.dtype,
            device=grad.device,
        )
    else:
        if tuple(out.shape) != tuple(sizes):
            raise ValueError("out shape mismatch")
        if out.dtype != grad.dtype:
            raise ValueError("dtype mismatch")
        if out.device != grad.device:
            raise ValueError("device mismatch")

        out.zero_()

    dim_stride = dim_size * inner_size

    BLOCK = 1024
    n_elements = outer_size * inner_size
    grid = (triton.cdiv(n_elements, BLOCK),)

    _cuda_select_backward_kernel[grid](
        grad_view,
        out,
        outer_size,
        inner_size,
        dim_stride,
        index,
        BLOCK=BLOCK,
    )

    return out


def select_backward(grad, input_sizes, dim, index, out=None):
    if _is_cuda_device(grad):
        return _cuda_launch_select_backward(grad, input_sizes, dim, index, out=out)
    return _launch_select_backward(grad, input_sizes, dim, index, out=out)