import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

_UNIQUE_DIM_COMPARE_BLOCK_SIZE = 1024
_UNIQUE_DIM_GATHER_BLOCK_SIZE = 1024
_UNIQUE_DIM_GROUP_SCAN_BLOCK_SIZE = 4096
_UNIQUE_DIM_RANK_SORT_MAX_KEYS = 4096
_UNIQUE_DIM_RADIX_BLOCK_SIZE = 512
_UNIQUE_DIM_RADIX_BITS = 4
_UNIQUE_DIM_HASH_MIN_ROW_LEN = 1024
_UNIQUE_DIM_FUSED_ALL_UNIQUE_MAX_ELEMENTS = 1 << 20
_UNIQUE_DIM_STRIDED_DIM1_FAST_MAX_ELEMENTS = 1 << 20
_UNIQUE_DIM_ASCEND_PREFIX_RANK_MAX_KEYS = 1024


# Per-column bit budgets and to-int64 conversions that preserve the original
# value ordering. The encodings let us pack a per-row ``group_id`` together
# with a single column's key into one int64 that, when compared as signed
# int64, matches the lex order over ``(group_id, signed_value)``.
_INT_DTYPE_BITS = {
    torch.bool: 1,
    torch.int8: 8,
    torch.uint8: 8,
    torch.int16: 16,
    torch.int32: 32,
    torch.float16: 16,
    torch.bfloat16: 16,
    torch.float32: 32,
}


@libentry()
@triton.jit
def _unique_dim_argsort_rank_kernel(
    keys_ptr: tl.tensor,
    indices_ptr: tl.tensor,
    sorted_keys_ptr: tl.tensor,
    num_keys: int,
    BLOCK_SIZE: tl.constexpr,
    STORE_SORTED_KEYS: tl.constexpr,
):
    row = ext.program_id(0)
    candidates = tl.arange(0, BLOCK_SIZE)
    mask = candidates < num_keys

    cur = tl.load(keys_ptr + row)
    vals = tl.load(keys_ptr + candidates, mask=mask, other=cur)
    before = ((vals < cur) | ((vals == cur) & (candidates < row))) & mask
    rank = tl.sum(before.to(tl.int32), axis=0)
    tl.store(indices_ptr + rank, row)
    if STORE_SORTED_KEYS:
        tl.store(sorted_keys_ptr + rank, cur)


@libentry()
@triton.jit
def _unique_dim_first_col_rank_kernel(
    flat_ptr: tl.tensor,
    indices_ptr: tl.tensor,
    group_id_ptr: tl.tensor,
    duplicate_flags_ptr: tl.tensor,
    num_rows: int,
    row_len: int,
    row_stride: int,
    KEY_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = ext.program_id(0)
    candidates = tl.arange(0, BLOCK_SIZE)
    mask = candidates < num_rows

    cur = tl.load(flat_ptr + row * row_stride).to(tl.int64) + KEY_OFFSET
    vals = (
        tl.load(flat_ptr + candidates * row_stride, mask=mask, other=0).to(tl.int64)
        + KEY_OFFSET
    )
    less = (vals < cur) & mask
    before = less | ((vals == cur) & (candidates < row) & mask)
    rank = tl.sum(before.to(tl.int32), axis=0)
    group_id = tl.sum(less.to(tl.int64), axis=0)
    tl.store(indices_ptr + rank, row)
    tl.store(group_id_ptr + rank, group_id)
    same_value = (vals == cur) & (candidates != row) & mask
    tl.store(duplicate_flags_ptr + row, tl.sum(same_value.to(tl.int32), axis=0) != 0)


@libentry()
@triton.jit
def _unique_dim_two_col_rank_kernel(
    flat_ptr: tl.tensor,
    indices_ptr: tl.tensor,
    group_id_ptr: tl.tensor,
    duplicate_flags_ptr: tl.tensor,
    num_rows: int,
    row_len: int,
    row_stride: int,
    col_stride: int,
    KEY_OFFSET: tl.constexpr,
    KEY_SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = ext.program_id(0)
    candidates = tl.arange(0, BLOCK_SIZE)
    mask = candidates < num_rows

    cur_base = row * row_stride
    cur0 = tl.load(flat_ptr + cur_base).to(tl.int64) + KEY_OFFSET
    cur1 = tl.load(flat_ptr + cur_base + col_stride).to(tl.int64) + KEY_OFFSET
    cur = cur0 * KEY_SCALE + cur1

    cand_base = candidates * row_stride
    vals0 = tl.load(flat_ptr + cand_base, mask=mask, other=0).to(tl.int64)
    vals1 = tl.load(
        flat_ptr + cand_base + col_stride,
        mask=mask,
        other=0,
    ).to(tl.int64)
    vals = (vals0 + KEY_OFFSET) * KEY_SCALE + (vals1 + KEY_OFFSET)

    less = (vals < cur) & mask
    before = less | ((vals == cur) & (candidates < row) & mask)
    rank = tl.sum(before.to(tl.int32), axis=0)
    group_id = tl.sum(less.to(tl.int64), axis=0)
    tl.store(indices_ptr + rank, row)
    tl.store(group_id_ptr + rank, group_id)
    same_value = (vals == cur) & (candidates != row) & mask
    tl.store(duplicate_flags_ptr + row, tl.sum(same_value.to(tl.int32), axis=0) != 0)


@libentry()
@triton.jit
def _unique_dim_radix_hist_kernel(
    keys_ptr: tl.tensor,
    hist_ptr: tl.tensor,
    num_keys: int,
    bit_offset: int,
    BLOCK_SIZE: tl.constexpr,
    RADIX_SIZE: tl.constexpr,
    RADIX_MASK: tl.constexpr,
):
    block = ext.program_id(0)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_keys

    keys = tl.load(keys_ptr + offsets, mask=mask, other=0)
    digits = ((keys >> bit_offset) & RADIX_MASK).to(tl.int32)
    for bin_id in tl.static_range(0, RADIX_SIZE):
        matches = (digits == bin_id) & mask
        count = tl.sum(matches.to(tl.int64), axis=0)
        tl.store(hist_ptr + block * RADIX_SIZE + bin_id, count)


@libentry()
@triton.jit
def _unique_dim_radix_prefix_kernel(
    hist_ptr: tl.tensor,
    block_prefix_ptr: tl.tensor,
    bin_counts_ptr: tl.tensor,
    num_blocks: int,
    BLOCK_BLOCKS: tl.constexpr,
    RADIX_SIZE: tl.constexpr,
):
    bin_id = ext.program_id(0)
    blocks = tl.arange(0, BLOCK_BLOCKS)
    mask = blocks < num_blocks

    counts = tl.load(
        hist_ptr + blocks * RADIX_SIZE + bin_id,
        mask=mask,
        other=0,
    )
    block_prefix = tl.cumsum(counts, axis=0) - counts
    tl.store(
        block_prefix_ptr + blocks * RADIX_SIZE + bin_id,
        block_prefix,
        mask=mask,
    )
    tl.store(bin_counts_ptr + bin_id, tl.sum(counts, axis=0))


@libentry()
@triton.jit
def _unique_dim_radix_scatter_kernel(
    keys_in_ptr: tl.tensor,
    indices_in_ptr: tl.tensor,
    keys_out_ptr: tl.tensor,
    indices_out_ptr: tl.tensor,
    block_prefix_ptr: tl.tensor,
    bin_counts_ptr: tl.tensor,
    num_keys: int,
    bit_offset: int,
    BLOCK_SIZE: tl.constexpr,
    RADIX_SIZE: tl.constexpr,
    RADIX_MASK: tl.constexpr,
):
    block = ext.program_id(0)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_keys

    keys = tl.load(keys_in_ptr + offsets, mask=mask, other=0)
    indices = tl.load(indices_in_ptr + offsets, mask=mask, other=0)
    digits = ((keys >> bit_offset) & RADIX_MASK).to(tl.int32)
    bins = tl.arange(0, RADIX_SIZE)
    bin_counts = tl.load(bin_counts_ptr + bins)
    bin_offsets = tl.cumsum(bin_counts, axis=0) - bin_counts

    for bin_id in tl.static_range(0, RADIX_SIZE):
        matches = (digits == bin_id) & mask
        local_rank = tl.cumsum(matches.to(tl.int64), axis=0) - matches.to(tl.int64)
        block_prefix = tl.load(block_prefix_ptr + block * RADIX_SIZE + bin_id)
        bin_offset = tl.sum(tl.where(bins == bin_id, bin_offsets, 0), axis=0)
        positions = bin_offset + block_prefix + local_rank
        tl.store(keys_out_ptr + positions, keys, mask=matches)
        tl.store(indices_out_ptr + positions, indices, mask=matches)


@libentry()
@triton.jit
def _unique_dim_gather_1d_kernel(
    input_ptr: tl.tensor,
    index_ptr: tl.tensor,
    output_ptr: tl.tensor,
    num_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    indices = tl.load(index_ptr + offsets, mask=mask, other=0)
    values = tl.load(input_ptr + indices, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)


@libentry()
@triton.jit
def _unique_dim_group_id_kernel(
    composite_ptr: tl.tensor,
    group_id_ptr: tl.tensor,
    last_group_id_ptr: tl.tensor,
    num_rows: int,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows
    cur = tl.load(composite_ptr + offsets, mask=mask, other=0)
    prev_offsets = tl.where(offsets == 0, 0, offsets - 1)
    prev = tl.load(composite_ptr + prev_offsets, mask=offsets > 0, other=cur)
    diff = ((cur - prev) != 0) & mask
    diff = tl.where(offsets == 0, False, diff)
    group_id = tl.cumsum(diff.to(tl.int64), axis=0)
    tl.store(group_id_ptr + offsets, group_id, mask=mask)
    last = tl.sum(tl.where(offsets == num_rows - 1, group_id, 0), axis=0)
    tl.store(last_group_id_ptr, last)


@libentry()
@triton.jit
def _unique_dim_any_bool_kernel(
    values_ptr: tl.tensor,
    any_ptr: tl.tensor,
    num_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    values = tl.load(values_ptr + offsets, mask=mask, other=0)
    any_value = tl.sum(values.to(tl.int32), axis=0) != 0
    tl.store(any_ptr, any_value)


@libentry()
@triton.jit
def _unique_dim_row_hash_chunk_kernel(
    flat_ptr: tl.tensor,
    chunk_hash_ptr: tl.tensor,
    num_rows: int,
    row_len: int,
    num_chunks: int,
    BLOCK_SIZE: tl.constexpr,
):
    row = ext.program_id(0)
    chunk = ext.program_id(1)
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_len

    vals = tl.load(flat_ptr + row * row_len + offsets, mask=mask, other=0)
    vals_i64 = vals.to(tl.int64)
    offsets_i64 = offsets.to(tl.int64)
    mix = (vals_i64 + (offsets_i64 + 1) * 1009 + 9176) * 131071
    mix = tl.where(mask, mix, 0)
    tl.store(chunk_hash_ptr + row * num_chunks + chunk, tl.sum(mix, axis=0))


@libentry()
@triton.jit
def _unique_dim_row_hash_reduce_kernel(
    chunk_hash_ptr: tl.tensor,
    row_hash_ptr: tl.tensor,
    num_chunks: int,
    BLOCK_CHUNKS: tl.constexpr,
):
    row = ext.program_id(0)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = chunks < num_chunks
    vals = tl.load(chunk_hash_ptr + row * num_chunks + chunks, mask=mask, other=0)
    tl.store(row_hash_ptr + row, tl.sum(vals, axis=0))


@libentry()
@triton.jit
def _unique_dim_row_chunk_diff_kernel(
    flat_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    row_chunk_diff_ptr: tl.tensor,
    num_rows: int,
    row_len: int,
    num_chunks: int,
    BLOCK_SIZE: tl.constexpr,
):
    row = ext.program_id(0)
    chunk = ext.program_id(1)
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_len

    out = tl.full((), 0, dtype=tl.int32)
    if row == 0:
        out = tl.where(chunk == 0, 1, 0)
    else:
        cur_row = tl.load(sorted_indices_ptr + row)
        prev_row = tl.load(sorted_indices_ptr + row - 1)
        cur = tl.load(flat_ptr + cur_row * row_len + offsets, mask=mask)
        prev = tl.load(flat_ptr + prev_row * row_len + offsets, mask=mask)
        neq = (cur != prev) & mask
        has_diff = tl.sum(neq.to(tl.int32), axis=0) != 0
        out = has_diff.to(tl.int32)
    tl.store(row_chunk_diff_ptr + row * num_chunks + chunk, out)


@libentry()
@triton.jit
def _unique_dim_row_chunk_diff_hash_kernel(
    flat_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    row_hash_ptr: tl.tensor,
    row_chunk_diff_ptr: tl.tensor,
    num_rows: int,
    row_len: int,
    num_chunks: int,
    BLOCK_SIZE: tl.constexpr,
):
    row = ext.program_id(0)
    chunk = ext.program_id(1)
    offsets = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_len

    out = tl.full((), 0, dtype=tl.int32)
    if row == 0:
        out = tl.where(chunk == 0, 1, 0)
    else:
        cur_row = tl.load(sorted_indices_ptr + row)
        prev_row = tl.load(sorted_indices_ptr + row - 1)
        cur_hash = tl.load(row_hash_ptr + cur_row)
        prev_hash = tl.load(row_hash_ptr + prev_row)
        if cur_hash != prev_hash:
            out = tl.where(chunk == 0, 1, 0)
        else:
            cur = tl.load(flat_ptr + cur_row * row_len + offsets, mask=mask)
            prev = tl.load(flat_ptr + prev_row * row_len + offsets, mask=mask)
            neq = (cur != prev) & mask
            has_diff = tl.sum(neq.to(tl.int32), axis=0) != 0
            out = has_diff.to(tl.int32)
    tl.store(row_chunk_diff_ptr + row * num_chunks + chunk, out)


@libentry()
@triton.jit
def _unique_dim_row_diff_reduce_kernel(
    row_chunk_diff_ptr: tl.tensor,
    is_first_ptr: tl.tensor,
    num_chunks: int,
    BLOCK_CHUNKS: tl.constexpr,
):
    row = ext.program_id(0)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = chunks < num_chunks
    vals = tl.load(row_chunk_diff_ptr + row * num_chunks + chunks, mask=mask, other=0)
    tl.store(is_first_ptr + row, tl.sum(vals, axis=0) != 0)


@libentry()
@triton.jit
def _unique_dim_row_single_chunk_first_kernel(
    flat_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    is_first_ptr: tl.tensor,
    num_rows: int,
    row_len: int,
    BLOCK_SIZE: tl.constexpr,
):
    row = ext.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_len

    out = tl.full((), True, dtype=tl.int1)
    if row != 0:
        cur_row = tl.load(sorted_indices_ptr + row)
        prev_row = tl.load(sorted_indices_ptr + row - 1)
        cur = tl.load(flat_ptr + cur_row * row_len + offsets, mask=mask)
        prev = tl.load(flat_ptr + prev_row * row_len + offsets, mask=mask)
        neq = (cur != prev) & mask
        out = tl.sum(neq.to(tl.int32), axis=0) != 0
    tl.store(is_first_ptr + row, out)


@libentry()
@triton.jit
def _unique_dim_gather_moved_kernel(
    flat_ptr: tl.tensor,
    unique_indices_ptr: tl.tensor,
    output_ptr: tl.tensor,
    total_elements: int,
    row_len: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    unique_pos = offsets // row_len
    col = offsets - unique_pos * row_len
    src_row = tl.load(unique_indices_ptr + unique_pos, mask=mask)
    values = tl.load(flat_ptr + src_row * row_len + col, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)


@libentry()
@triton.jit
def _unique_dim_gather_inverse_moved_kernel(
    flat_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    output_ptr: tl.tensor,
    inverse_ptr: tl.tensor,
    total_elements: int,
    row_len: int,
    BLOCK_SIZE: tl.constexpr,
    STORE_INVERSE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    sorted_pos = offsets // row_len
    col = offsets - sorted_pos * row_len
    src_row = tl.load(sorted_indices_ptr + sorted_pos, mask=mask)
    values = tl.load(flat_ptr + src_row * row_len + col, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)
    if STORE_INVERSE:
        tl.store(
            inverse_ptr + src_row,
            sorted_pos.to(tl.int64),
            mask=mask & (col == 0),
        )


@libentry()
@triton.jit
def _unique_dim_gather_inverse_strided_2d_kernel(
    input_ptr: tl.tensor,
    sorted_indices_ptr: tl.tensor,
    output_ptr: tl.tensor,
    inverse_ptr: tl.tensor,
    rows: int,
    cols: int,
    BLOCK_SIZE: tl.constexpr,
    STORE_INVERSE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = rows * cols
    mask = offsets < total_elements

    row = offsets // cols
    sorted_col = offsets - row * cols
    src_col = tl.load(sorted_indices_ptr + sorted_col, mask=mask)
    values = tl.load(input_ptr + row * cols + src_col, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)
    if STORE_INVERSE:
        tl.store(
            inverse_ptr + src_col,
            sorted_col.to(tl.int64),
            mask=mask & (row == 0),
        )


@libentry()
@triton.jit
def _unique_dim_inverse_kernel(
    sorted_indices_ptr: tl.tensor,
    inverse_sorted_ptr: tl.tensor,
    inverse_ptr: tl.tensor,
    num_rows: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows
    sorted_indices = tl.load(sorted_indices_ptr + offsets, mask=mask)
    inverse_sorted = tl.load(inverse_sorted_ptr + offsets, mask=mask)
    tl.store(inverse_ptr + sorted_indices, inverse_sorted, mask=mask)


@libentry()
@triton.jit
def _unique_dim_inverse_permutation_kernel(
    sorted_indices_ptr: tl.tensor,
    inverse_ptr: tl.tensor,
    num_rows: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows
    sorted_indices = tl.load(sorted_indices_ptr + offsets, mask=mask, other=0)
    tl.store(inverse_ptr + sorted_indices, offsets.to(tl.int64), mask=mask)


@libentry()
@triton.jit
def _unique_dim_inverse_small_kernel(
    sorted_indices_ptr: tl.tensor,
    is_first_ptr: tl.tensor,
    inverse_ptr: tl.tensor,
    num_rows: int,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows
    is_first = tl.load(is_first_ptr + offsets, mask=mask, other=0).to(tl.int64)
    inverse_sorted = tl.cumsum(is_first, axis=0) - 1
    sorted_indices = tl.load(sorted_indices_ptr + offsets, mask=mask, other=0)
    tl.store(inverse_ptr + sorted_indices, inverse_sorted, mask=mask)


@libentry()
@triton.jit
def _unique_dim_unique_indices_small_kernel(
    sorted_indices_ptr: tl.tensor,
    is_first_ptr: tl.tensor,
    unique_indices_ptr: tl.tensor,
    num_unique_ptr: tl.tensor,
    num_rows: int,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows
    is_first = tl.load(is_first_ptr + offsets, mask=mask, other=0).to(tl.int64)
    positions = tl.cumsum(is_first, axis=0) - 1
    sorted_indices = tl.load(sorted_indices_ptr + offsets, mask=mask, other=0)
    tl.store(
        unique_indices_ptr + positions,
        sorted_indices,
        mask=mask & (is_first != 0),
    )
    num_unique = tl.sum(tl.where(mask, is_first, 0), axis=0)
    tl.store(num_unique_ptr, num_unique)


@libentry()
@triton.jit
def _unique_dim_unique_indices_inverse_small_kernel(
    sorted_indices_ptr: tl.tensor,
    is_first_ptr: tl.tensor,
    unique_indices_ptr: tl.tensor,
    inverse_ptr: tl.tensor,
    num_unique_ptr: tl.tensor,
    num_rows: int,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_rows
    is_first = tl.load(is_first_ptr + offsets, mask=mask, other=0).to(tl.int64)
    positions = tl.cumsum(is_first, axis=0) - 1
    sorted_indices = tl.load(sorted_indices_ptr + offsets, mask=mask, other=0)
    tl.store(
        unique_indices_ptr + positions,
        sorted_indices,
        mask=mask & (is_first != 0),
    )
    tl.store(inverse_ptr + sorted_indices, positions, mask=mask)
    num_unique = tl.sum(tl.where(mask, is_first, 0), axis=0)
    tl.store(num_unique_ptr, num_unique)


@libentry()
@triton.jit
def _unique_dim_counts_kernel(
    first_positions_ptr: tl.tensor,
    counts_ptr: tl.tensor,
    num_rows: int,
    num_unique: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_unique
    positions = tl.load(first_positions_ptr + offsets, mask=mask)
    next_positions = tl.load(
        first_positions_ptr + offsets + 1,
        mask=(offsets + 1) < num_unique,
        other=num_rows,
    )
    tl.store(counts_ptr + offsets, next_positions - positions, mask=mask)


def _triton_num_warps(block_size: int) -> int:
    if block_size >= 8192:
        return 8
    if block_size >= 2048:
        return 4
    return 1


def _unique_dim_radix_num_passes(keys: torch.Tensor) -> int:
    max_key = keys.max().item()
    return max(1, triton.cdiv(max(1, int(max_key).bit_length()), _UNIQUE_DIM_RADIX_BITS))


def _monotonic_key_bits(dtype: torch.dtype):
    """Return the per-element key width for ``dtype`` if it can be mapped
    into a monotonic int64 view, else ``None``."""
    return _INT_DTYPE_BITS.get(dtype)


def _monotonic_int64_column(flat: torch.Tensor, col: int) -> torch.Tensor:
    """Apply the dtype-appropriate monotonic remap to a single column of a
    ``(D, M)`` tensor and return a fresh ``(D,)`` int64 tensor.

    Computing per-column avoids materializing the full ``(D, M)`` int64
    tensor (which costs ``8 * D * M`` bytes) for wide inputs.
    """
    dt = flat.dtype
    col_data = flat[:, col].contiguous()
    if dt in (torch.uint8, torch.bool):
        return col_data.to(torch.int64)
    if dt == torch.int8:
        return col_data.to(torch.int64) + (1 << 7)
    if dt == torch.int16:
        return col_data.to(torch.int64) + (1 << 15)
    if dt == torch.int32:
        return col_data.to(torch.int64) + (1 << 31)
    if dt in (torch.float16, torch.bfloat16):
        as_int = col_data.view(torch.int16).to(torch.int64) & 0xFFFF
        sign_set = (as_int & 0x8000) != 0
        return torch.where(sign_set, as_int ^ 0xFFFF, as_int ^ 0x8000)
    if dt == torch.float32:
        as_int = col_data.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        sign_set = (as_int & 0x80000000) != 0
        return torch.where(
            sign_set, as_int ^ 0xFFFFFFFF, as_int ^ 0x80000000
        )
    raise NotImplementedError(dt)


def _first_col_key_offset(dtype: torch.dtype):
    if dtype == torch.int16:
        return 1 << 15
    if dtype == torch.int32:
        return 1 << 31
    return None


def _triton_first_col_argsort(
    values: torch.Tensor,
    num_rows: int,
    row_len: int,
    row_stride: int,
):
    key_offset = _first_col_key_offset(values.dtype)
    if (
        key_offset is None
        or row_len == 0
        or num_rows == 0
        or num_rows > _UNIQUE_DIM_RANK_SORT_MAX_KEYS
        or (
            values.device.type != "cuda"
            and num_rows > _UNIQUE_DIM_ASCEND_PREFIX_RANK_MAX_KEYS
        )
    ):
        return None

    indices = torch.empty(num_rows, dtype=torch.int64, device=values.device)
    group_id = torch.empty(num_rows, dtype=torch.int64, device=values.device)
    duplicate_flags = torch.empty(num_rows, dtype=torch.bool, device=values.device)
    block_size = triton.next_power_of_2(num_rows)
    with torch_device_fn.device(values.device.index):
        _unique_dim_first_col_rank_kernel[(num_rows, 1, 1)](
            values,
            indices,
            group_id,
            duplicate_flags,
            num_rows,
            row_len,
            row_stride,
            KEY_OFFSET=key_offset,
            BLOCK_SIZE=block_size,
            num_warps=_triton_num_warps(block_size),
        )
    return indices, group_id, duplicate_flags


def _triton_two_col_argsort_int16(
    values: torch.Tensor,
    num_rows: int,
    row_len: int,
    row_stride: int,
    col_stride: int,
):
    if (
        values.device.type != "cuda"
        or values.dtype != torch.int16
        or row_len < 2
        or num_rows == 0
        or num_rows > _UNIQUE_DIM_RANK_SORT_MAX_KEYS
    ):
        return None

    indices = torch.empty(num_rows, dtype=torch.int64, device=values.device)
    group_id = torch.empty(num_rows, dtype=torch.int64, device=values.device)
    duplicate_flags = torch.empty(num_rows, dtype=torch.bool, device=values.device)
    block_size = triton.next_power_of_2(num_rows)
    with torch_device_fn.device(values.device.index):
        _unique_dim_two_col_rank_kernel[(num_rows, 1, 1)](
            values,
            indices,
            group_id,
            duplicate_flags,
            num_rows,
            row_len,
            row_stride,
            col_stride,
            KEY_OFFSET=1 << 15,
            KEY_SCALE=1 << 16,
            BLOCK_SIZE=block_size,
            num_warps=_triton_num_warps(block_size),
        )
    return indices, group_id, duplicate_flags


def _triton_prefix_argsort(
    values: torch.Tensor,
    num_rows: int,
    row_len: int,
    row_stride: int,
    col_stride: int,
):
    two_col_sort = _triton_two_col_argsort_int16(
        values,
        num_rows,
        row_len,
        row_stride,
        col_stride,
    )
    if two_col_sort is not None:
        return *two_col_sort, 2

    first_col_sort = _triton_first_col_argsort(values, num_rows, row_len, row_stride)
    if first_col_sort is not None:
        return *first_col_sort, 1
    return None


def _triton_any_bool(values: torch.Tensor) -> torch.Tensor:
    any_value = torch.empty((), dtype=torch.bool, device=values.device)
    num_elements = values.numel()
    if num_elements == 0:
        any_value.fill_(False)
        return any_value
    block_size = triton.next_power_of_2(num_elements)
    with torch_device_fn.device(values.device.index):
        _unique_dim_any_bool_kernel[(1, 1, 1)](
            values,
            any_value,
            num_elements,
            BLOCK_SIZE=block_size,
            num_warps=_triton_num_warps(block_size),
        )
    return any_value


def _triton_gather_1d(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    num_elements = indices.numel()
    output = torch.empty(num_elements, dtype=values.dtype, device=values.device)
    if num_elements == 0:
        return output
    grid = (triton.cdiv(num_elements, _UNIQUE_DIM_GATHER_BLOCK_SIZE), 1, 1)
    with torch_device_fn.device(values.device.index):
        _unique_dim_gather_1d_kernel[grid](
            values,
            indices,
            output,
            num_elements,
            BLOCK_SIZE=_UNIQUE_DIM_GATHER_BLOCK_SIZE,
            num_warps=4,
        )
    return output


def _triton_group_id_from_sorted_composite(composite: torch.Tensor):
    num_rows = composite.numel()
    group_id = torch.empty(num_rows, dtype=torch.int64, device=composite.device)
    last_group_id = torch.empty((), dtype=torch.int64, device=composite.device)
    if num_rows == 0:
        return group_id, last_group_id

    block_size = triton.next_power_of_2(num_rows)
    with torch_device_fn.device(composite.device.index):
        _unique_dim_group_id_kernel[(1, 1, 1)](
            composite,
            group_id,
            last_group_id,
            num_rows,
            BLOCK_SIZE=block_size,
            num_warps=_triton_num_warps(block_size),
        )
    return group_id, last_group_id


def _lex_argsort_rows_composite(flat: torch.Tensor):
    """Lex-sort rows by packing ``(group_id, monotonic_key)`` per column.

    Mirrors the way ATen's CUDA ``unique_dim`` does a single comparator-driven
    sort: each cascade step performs *one* ``argsort`` on an int64 key that
    encodes "current lex prefix" in the high bits and "this column's value"
    in the low bits. As soon as every row has a unique prefix we terminate;
    for random data this happens after one or two columns even when ``M``
    is large, replacing ``M`` argsorts with a small constant.
    """
    key_bits = _monotonic_key_bits(flat.dtype)
    if key_bits is None:
        return None

    num_rows, num_cols = flat.shape
    device = flat.device
    if num_cols == 0:
        indices = torch.arange(num_rows, dtype=torch.int64, device=device)
        return indices, False
    if num_rows <= 1:
        indices = torch.arange(num_rows, dtype=torch.int64, device=device)
        return indices, True

    key_scale = 1 << key_bits
    all_unique = False
    prefix_sort = _triton_prefix_argsort(
        flat,
        num_rows,
        num_cols,
        row_stride=num_cols,
        col_stride=1,
    )
    if prefix_sort is None:
        indices = torch.arange(num_rows, dtype=torch.int64, device=device)
        group_id = torch.zeros(num_rows, dtype=torch.int64, device=device)
        start_col = 0
    else:
        indices, group_id, duplicate_flags, start_col = prefix_sort
        has_duplicate = _triton_any_bool(duplicate_flags)
        if not has_duplicate.item():
            return indices, True

    for col in range(start_col, num_cols):
        column_keys = _monotonic_int64_column(flat, col)
        keys = column_keys if col == 0 else _triton_gather_1d(column_keys, indices)
        # Use ``group_id * scale + keys`` rather than
        # ``(group_id << bits) | keys``. Functionally identical because
        # ``keys`` is in ``[0, scale)`` after the monotonic remap, but the
        # multiply/add path avoids the int64 bitwise kernels that some
        # Ascend/NPU backends do not provide.
        composite = group_id * key_scale + keys
        perm, composite = _triton_argsort_1d(
            composite,
            nonnegative_int64=True,
            return_sorted=True,
        )
        indices = perm if col == 0 else _triton_gather_1d(indices, perm)
        # When running under FlagGems' op interception, the registered int64
        # tensor-vs-tensor comparison ops (and the bool dtype cast) route
        # through float32 and lose precision around 2**24, silently mapping
        # non-equal composite values to ``False``. ``int64 - int64`` followed
        # by tensor-vs-scalar ``ne 0`` is the path that stays exact.
        if num_rows <= _UNIQUE_DIM_GROUP_SCAN_BLOCK_SIZE:
            group_id, last_group_id = _triton_group_id_from_sorted_composite(composite)
            # Early termination: every row has a unique lex prefix already.
            if last_group_id.item() == num_rows - 1:
                all_unique = True
                break
        else:
            diff = ((composite[1:] - composite[:-1]) != 0).to(torch.int64)
            group_id = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int64, device=device),
                    torch.cumsum(diff, dim=0),
                ]
            )
            # Early termination: every row has a unique lex prefix already.
            if group_id[-1].item() == num_rows - 1:
                all_unique = True
                break
    return indices, all_unique


def _triton_radix_argsort_nonnegative_int64(
    keys: torch.Tensor,
    return_sorted: bool = False,
):
    """Stable LSD radix argsort for non-negative int64 composite keys."""
    num_keys = keys.numel()
    indices = torch.arange(num_keys, dtype=torch.int64, device=keys.device)
    if num_keys <= 1:
        return (indices, keys) if return_sorted else indices

    keys_in = torch.clone(keys.contiguous())
    keys_out = torch.empty_like(keys_in)
    indices_in = indices
    indices_out = torch.empty_like(indices_in)

    radix_size = 1 << _UNIQUE_DIM_RADIX_BITS
    radix_mask = radix_size - 1
    num_blocks = triton.cdiv(num_keys, _UNIQUE_DIM_RADIX_BLOCK_SIZE)
    hist = torch.empty((num_blocks, radix_size), dtype=torch.int64, device=keys.device)
    block_prefix = torch.empty_like(hist)
    bin_counts = torch.empty(radix_size, dtype=torch.int64, device=keys.device)
    prefix_block_size = triton.next_power_of_2(num_blocks)
    num_passes = _unique_dim_radix_num_passes(keys_in)

    with torch_device_fn.device(keys.device.index):
        for pass_id in range(num_passes):
            bit_offset = pass_id * _UNIQUE_DIM_RADIX_BITS
            _unique_dim_radix_hist_kernel[(num_blocks, 1, 1)](
                keys_in,
                hist,
                num_keys,
                bit_offset,
                BLOCK_SIZE=_UNIQUE_DIM_RADIX_BLOCK_SIZE,
                RADIX_SIZE=radix_size,
                RADIX_MASK=radix_mask,
                num_warps=4,
            )
            _unique_dim_radix_prefix_kernel[(radix_size, 1, 1)](
                hist,
                block_prefix,
                bin_counts,
                num_blocks,
                BLOCK_BLOCKS=prefix_block_size,
                RADIX_SIZE=radix_size,
                num_warps=_triton_num_warps(prefix_block_size),
            )
            _unique_dim_radix_scatter_kernel[(num_blocks, 1, 1)](
                keys_in,
                indices_in,
                keys_out,
                indices_out,
                block_prefix,
                bin_counts,
                num_keys,
                bit_offset,
                BLOCK_SIZE=_UNIQUE_DIM_RADIX_BLOCK_SIZE,
                RADIX_SIZE=radix_size,
                RADIX_MASK=radix_mask,
                num_warps=4,
            )
            keys_in, keys_out = keys_out, keys_in
            indices_in, indices_out = indices_out, indices_in
    if return_sorted:
        return indices_in, keys_in
    return indices_in


def _triton_argsort_1d(
    keys: torch.Tensor,
    nonnegative_int64: bool = False,
    return_sorted: bool = False,
) -> torch.Tensor:
    """Stable ascending argsort. Composite int64 keys use a Triton radix path."""
    num_keys = keys.numel()
    indices = torch.empty(num_keys, dtype=torch.int64, device=keys.device)
    if num_keys == 0:
        return (indices, keys) if return_sorted else indices
    if (
        nonnegative_int64
        and keys.dtype == torch.int64
        and num_keys > _UNIQUE_DIM_RANK_SORT_MAX_KEYS
    ):
        return _triton_radix_argsort_nonnegative_int64(
            keys,
            return_sorted=return_sorted,
        )

    block_size = triton.next_power_of_2(num_keys)
    sorted_keys = torch.empty_like(keys) if return_sorted else keys
    with torch_device_fn.device(keys.device.index):
        _unique_dim_argsort_rank_kernel[(num_keys, 1, 1)](
            keys.contiguous(),
            indices,
            sorted_keys,
            num_keys,
            BLOCK_SIZE=block_size,
            STORE_SORTED_KEYS=return_sorted,
            num_warps=_triton_num_warps(block_size),
        )
    if return_sorted:
        return indices, sorted_keys
    return indices


def _lex_argsort_rows_cascade(flat: torch.Tensor) -> torch.Tensor:
    """Generic-dtype fallback: cascade of stable argsorts, least to most
    significant column. ``O(M)`` argsorts of length ``D`` with ``O(D)`` memory
    traffic per step."""
    num_rows, num_cols = flat.shape
    indices = torch.arange(num_rows, dtype=torch.int64, device=flat.device)
    if num_rows <= 1 or num_cols == 0:
        return indices
    flat_t = flat.t().contiguous()
    for col in range(num_cols - 1, -1, -1):
        keys = _triton_gather_1d(flat_t[col], indices)
        perm = _triton_argsort_1d(keys)
        indices = _triton_gather_1d(indices, perm)
    return indices


def _lex_argsort_rows(flat: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return indices that sort rows of a 2D tensor lexicographically."""
    composite = _lex_argsort_rows_composite(flat)
    if composite is not None:
        return composite
    return _lex_argsort_rows_cascade(flat), False


def _unique_dim_row_hash(flat: torch.Tensor) -> torch.Tensor:
    num_rows, row_len = flat.shape
    block_size = min(_UNIQUE_DIM_COMPARE_BLOCK_SIZE, triton.next_power_of_2(row_len))
    num_chunks = triton.cdiv(row_len, block_size)
    chunk_hash = torch.empty((num_rows, num_chunks), dtype=torch.int64, device=flat.device)
    row_hash = torch.empty(num_rows, dtype=torch.int64, device=flat.device)
    with torch_device_fn.device(flat.device.index):
        _unique_dim_row_hash_chunk_kernel[(num_rows, num_chunks, 1)](
            flat,
            chunk_hash,
            num_rows,
            row_len,
            num_chunks,
            BLOCK_SIZE=block_size,
            num_warps=_triton_num_warps(block_size),
        )
        _unique_dim_row_hash_reduce_kernel[(num_rows, 1, 1)](
            chunk_hash,
            row_hash,
            num_chunks,
            BLOCK_CHUNKS=triton.next_power_of_2(num_chunks),
            num_warps=_triton_num_warps(triton.next_power_of_2(num_chunks)),
        )
    return row_hash


def _unique_dim_first_mask(flat: torch.Tensor, sorted_indices: torch.Tensor):
    """Return a bool mask for first rows in sorted lexicographic groups."""
    num_rows, row_len = flat.shape
    if num_rows == 1 or row_len == 0:
        is_first = torch.zeros(num_rows, dtype=torch.bool, device=flat.device)
        is_first[0] = True
        return is_first

    block_size = min(_UNIQUE_DIM_COMPARE_BLOCK_SIZE, triton.next_power_of_2(row_len))
    num_chunks = triton.cdiv(row_len, block_size)
    is_first = torch.empty(num_rows, dtype=torch.bool, device=flat.device)
    if num_chunks == 1:
        with torch_device_fn.device(flat.device.index):
            _unique_dim_row_single_chunk_first_kernel[(num_rows, 1, 1)](
                flat,
                sorted_indices,
                is_first,
                num_rows,
                row_len,
                BLOCK_SIZE=block_size,
                num_warps=_triton_num_warps(block_size),
            )
        return is_first

    row_chunk_diff = torch.empty(
        (num_rows, num_chunks), dtype=torch.int32, device=flat.device
    )
    grid = (num_rows, num_chunks, 1)
    row_hash = (
        _unique_dim_row_hash(flat) if row_len >= _UNIQUE_DIM_HASH_MIN_ROW_LEN else None
    )
    with torch_device_fn.device(flat.device.index):
        if row_hash is None:
            _unique_dim_row_chunk_diff_kernel[grid](
                flat,
                sorted_indices,
                row_chunk_diff,
                num_rows,
                row_len,
                num_chunks,
                BLOCK_SIZE=block_size,
                num_warps=_triton_num_warps(block_size),
            )
        else:
            _unique_dim_row_chunk_diff_hash_kernel[grid](
                flat,
                sorted_indices,
                row_hash,
                row_chunk_diff,
                num_rows,
                row_len,
                num_chunks,
                BLOCK_SIZE=block_size,
                num_warps=_triton_num_warps(block_size),
            )
        _unique_dim_row_diff_reduce_kernel[(num_rows, 1, 1)](
            row_chunk_diff,
            is_first,
            num_chunks,
            BLOCK_CHUNKS=triton.next_power_of_2(num_chunks),
            num_warps=_triton_num_warps(triton.next_power_of_2(num_chunks)),
        )
    return is_first


def _unique_dim_gather_output(
    moved: torch.Tensor,
    unique_indices: torch.Tensor,
    dim: int,
    input_shape: torch.Size,
) -> torch.Tensor:
    num_unique = unique_indices.numel()
    output_shape = (
        tuple(input_shape[:dim]) + (num_unique,) + tuple(input_shape[dim + 1 :])
    )
    if num_unique == 0:
        return torch.empty(output_shape, dtype=moved.dtype, device=moved.device)

    row_len = moved[0].numel()
    flat = moved.reshape(moved.shape[0], row_len)
    moved_output = torch.empty(
        (num_unique,) + tuple(moved.shape[1:]),
        dtype=moved.dtype,
        device=moved.device,
    )
    grid = (triton.cdiv(moved_output.numel(), _UNIQUE_DIM_GATHER_BLOCK_SIZE), 1, 1)
    with torch_device_fn.device(moved.device.index):
        _unique_dim_gather_moved_kernel[grid](
            flat,
            unique_indices,
            moved_output,
            moved_output.numel(),
            row_len,
            BLOCK_SIZE=_UNIQUE_DIM_GATHER_BLOCK_SIZE,
            num_warps=4,
        )
    return moved_output.movedim(0, dim)


def _unique_dim_gather_all_unique(
    moved: torch.Tensor,
    sorted_indices: torch.Tensor,
    dim: int,
    input_shape: torch.Size,
    return_inverse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows = sorted_indices.numel()
    output_shape = (
        tuple(input_shape[:dim]) + (num_rows,) + tuple(input_shape[dim + 1 :])
    )
    inverse_indices = (
        torch.empty_like(sorted_indices)
        if return_inverse
        else torch.empty(0, dtype=torch.int64, device=moved.device)
    )
    if num_rows == 0:
        output = torch.empty(output_shape, dtype=moved.dtype, device=moved.device)
        return output, inverse_indices

    row_len = moved[0].numel()
    flat = moved.reshape(num_rows, row_len)
    moved_output = torch.empty(
        (num_rows,) + tuple(moved.shape[1:]),
        dtype=moved.dtype,
        device=moved.device,
    )
    grid = (triton.cdiv(moved_output.numel(), _UNIQUE_DIM_GATHER_BLOCK_SIZE), 1, 1)
    with torch_device_fn.device(moved.device.index):
        _unique_dim_gather_inverse_moved_kernel[grid](
            flat,
            sorted_indices,
            moved_output,
            inverse_indices,
            moved_output.numel(),
            row_len,
            BLOCK_SIZE=_UNIQUE_DIM_GATHER_BLOCK_SIZE,
            STORE_INVERSE=return_inverse,
            num_warps=4,
        )
    return moved_output.movedim(0, dim), inverse_indices


def _unique_dim_gather_all_unique_strided_2d(
    input: torch.Tensor,
    sorted_indices: torch.Tensor,
    return_inverse: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = input.shape
    inverse_indices = (
        torch.empty_like(sorted_indices)
        if return_inverse
        else torch.empty(0, dtype=torch.int64, device=input.device)
    )
    output = torch.empty_like(input)
    if rows == 0 or cols == 0:
        return output, inverse_indices

    grid = (triton.cdiv(output.numel(), _UNIQUE_DIM_GATHER_BLOCK_SIZE), 1, 1)
    with torch_device_fn.device(input.device.index):
        _unique_dim_gather_inverse_strided_2d_kernel[grid](
            input,
            sorted_indices,
            output,
            inverse_indices,
            rows,
            cols,
            BLOCK_SIZE=_UNIQUE_DIM_GATHER_BLOCK_SIZE,
            STORE_INVERSE=return_inverse,
            num_warps=4,
        )
    return output, inverse_indices


def _unique_dim_inverse_from_permutation(sorted_indices: torch.Tensor) -> torch.Tensor:
    num_rows = sorted_indices.numel()
    inverse_indices = torch.empty_like(sorted_indices)
    if num_rows == 0:
        return inverse_indices
    grid = (triton.cdiv(num_rows, _UNIQUE_DIM_GATHER_BLOCK_SIZE), 1, 1)
    with torch_device_fn.device(sorted_indices.device.index):
        _unique_dim_inverse_permutation_kernel[grid](
            sorted_indices,
            inverse_indices,
            num_rows,
            BLOCK_SIZE=_UNIQUE_DIM_GATHER_BLOCK_SIZE,
            num_warps=4,
        )
    return inverse_indices


def _unique_dim_inverse(
    sorted_indices: torch.Tensor,
    is_first: torch.Tensor,
) -> torch.Tensor:
    num_rows = sorted_indices.numel()
    inverse_indices = torch.empty(
        num_rows, dtype=torch.int64, device=sorted_indices.device
    )
    if num_rows <= _UNIQUE_DIM_GROUP_SCAN_BLOCK_SIZE:
        block_size = triton.next_power_of_2(num_rows)
        with torch_device_fn.device(sorted_indices.device.index):
            _unique_dim_inverse_small_kernel[(1, 1, 1)](
                sorted_indices,
                is_first,
                inverse_indices,
                num_rows,
                BLOCK_SIZE=block_size,
                num_warps=_triton_num_warps(block_size),
            )
        return inverse_indices

    inverse_in_sorted = torch.cumsum(is_first.to(torch.int64), dim=0) - 1
    grid = (triton.cdiv(num_rows, _UNIQUE_DIM_GATHER_BLOCK_SIZE), 1, 1)
    with torch_device_fn.device(sorted_indices.device.index):
        _unique_dim_inverse_kernel[grid](
            sorted_indices,
            inverse_in_sorted,
            inverse_indices,
            num_rows,
            BLOCK_SIZE=_UNIQUE_DIM_GATHER_BLOCK_SIZE,
            num_warps=4,
        )
    return inverse_indices


def _unique_dim_unique_indices(
    sorted_indices: torch.Tensor,
    is_first: torch.Tensor,
) -> torch.Tensor:
    num_rows = sorted_indices.numel()
    if num_rows <= _UNIQUE_DIM_GROUP_SCAN_BLOCK_SIZE:
        unique_indices_full = torch.empty_like(sorted_indices)
        num_unique_tensor = torch.empty(
            (), dtype=torch.int64, device=sorted_indices.device
        )
        block_size = triton.next_power_of_2(num_rows)
        with torch_device_fn.device(sorted_indices.device.index):
            _unique_dim_unique_indices_small_kernel[(1, 1, 1)](
                sorted_indices,
                is_first,
                unique_indices_full,
                num_unique_tensor,
                num_rows,
                BLOCK_SIZE=block_size,
                num_warps=_triton_num_warps(block_size),
            )
        return unique_indices_full[: num_unique_tensor.item()]

    return sorted_indices.masked_select(is_first)


def _unique_dim_unique_indices_and_inverse(
    sorted_indices: torch.Tensor,
    is_first: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows = sorted_indices.numel()
    if num_rows <= _UNIQUE_DIM_GROUP_SCAN_BLOCK_SIZE:
        unique_indices_full = torch.empty_like(sorted_indices)
        inverse_indices = torch.empty_like(sorted_indices)
        num_unique_tensor = torch.empty(
            (), dtype=torch.int64, device=sorted_indices.device
        )
        block_size = triton.next_power_of_2(num_rows)
        with torch_device_fn.device(sorted_indices.device.index):
            _unique_dim_unique_indices_inverse_small_kernel[(1, 1, 1)](
                sorted_indices,
                is_first,
                unique_indices_full,
                inverse_indices,
                num_unique_tensor,
                num_rows,
                BLOCK_SIZE=block_size,
                num_warps=_triton_num_warps(block_size),
            )
        return unique_indices_full[: num_unique_tensor.item()], inverse_indices

    unique_indices = sorted_indices.masked_select(is_first)
    inverse_indices = _unique_dim_inverse(sorted_indices, is_first)
    return unique_indices, inverse_indices


def _unique_dim_counts(
    is_first: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    first_positions = torch.nonzero(is_first, as_tuple=False).flatten()
    num_unique = first_positions.numel()
    counts = torch.empty(num_unique, dtype=torch.int64, device=is_first.device)
    if num_unique == 0:
        return counts

    grid = (triton.cdiv(num_unique, _UNIQUE_DIM_GATHER_BLOCK_SIZE), 1, 1)
    with torch_device_fn.device(is_first.device.index):
        _unique_dim_counts_kernel[grid](
            first_positions,
            counts,
            num_rows,
            num_unique,
            BLOCK_SIZE=_UNIQUE_DIM_GATHER_BLOCK_SIZE,
            num_warps=4,
        )
    return counts


def _unique_dim_dim1_2d_all_unique_fast_path(
    input: torch.Tensor,
    return_inverse: bool,
    return_counts: bool,
):
    if (
        input.device.type != "cuda"
        or input.ndim != 2
        or input.stride(1) != 1
        or input.dtype not in (torch.int16, torch.int32)
        or input.size(1) == 0
        or input.size(1) > _UNIQUE_DIM_RANK_SORT_MAX_KEYS
        or input.numel() > _UNIQUE_DIM_STRIDED_DIM1_FAST_MAX_ELEMENTS
    ):
        return None

    rows, cols = input.shape
    prefix_sort = _triton_prefix_argsort(
        input,
        cols,
        rows,
        row_stride=1,
        col_stride=cols,
    )
    if prefix_sort is None:
        return None

    sorted_indices, _, duplicate_flags, _ = prefix_sort
    has_duplicate = _triton_any_bool(duplicate_flags)
    if has_duplicate.item():
        return None

    output, inverse_indices = _unique_dim_gather_all_unique_strided_2d(
        input,
        sorted_indices,
        return_inverse,
    )
    counts = (
        torch.ones(cols, dtype=torch.int64, device=input.device)
        if return_counts
        else torch.empty(0, dtype=torch.int64, device=input.device)
    )
    return output, inverse_indices, counts


def unique_dim(
    input: torch.Tensor,
    dim: int,
    sorted: bool = True,
    return_inverse: bool = False,
    return_counts: bool = False,
):
    """Dimension-aware ``torch.unique`` (a.k.a. ``aten::unique_dim``).

    Treats each slice along ``dim`` as a single element, returning the unique
    slices, an optional inverse mapping of shape ``(input.size(dim),)`` and an
    optional per-unique count tensor of shape ``(output.size(dim),)``.
    """
    logger.debug("GEMS UNIQUE_DIM")

    ndim = input.ndim if input.ndim > 0 else 1
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= max(input.ndim, 1):
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-input.ndim}, {input.ndim - 1}], but got {dim})"
        )

    device = input.device
    size_dim = input.size(dim) if input.ndim > 0 else input.numel()

    if size_dim == 0:
        output = input.clone()
        inverse_indices = (
            torch.empty(0, dtype=torch.int64, device=device)
            if return_inverse
            else torch.empty(0, dtype=torch.int64, device=device)
        )
        counts = (
            torch.empty(0, dtype=torch.int64, device=device)
            if return_counts
            else torch.empty(0, dtype=torch.int64, device=device)
        )
        return output, inverse_indices, counts

    if dim == 1:
        fast_result = _unique_dim_dim1_2d_all_unique_fast_path(
            input,
            return_inverse,
            return_counts,
        )
        if fast_result is not None:
            return fast_result

    moved = input.movedim(dim, 0).contiguous()
    flat = moved.reshape(size_dim, -1)

    sorted_indices, all_unique = _lex_argsort_rows(flat)

    inverse_indices = torch.empty(0, dtype=torch.int64, device=device)
    counts = torch.empty(0, dtype=torch.int64, device=device)

    if all_unique:
        unique_in_orig = sorted_indices
        if return_counts:
            counts = torch.ones(size_dim, dtype=torch.int64, device=device)
        if (
            device.type == "cuda"
            and return_inverse
            and moved.numel() <= _UNIQUE_DIM_FUSED_ALL_UNIQUE_MAX_ELEMENTS
        ):
            output, inverse_indices = _unique_dim_gather_all_unique(
                moved,
                sorted_indices,
                dim,
                input.shape,
                return_inverse,
            )
        else:
            if return_inverse:
                inverse_indices = _unique_dim_inverse_from_permutation(sorted_indices)
            output = _unique_dim_gather_output(moved, unique_in_orig, dim, input.shape)
        return output, inverse_indices, counts
    else:
        is_first = _unique_dim_first_mask(flat, sorted_indices)
        if return_inverse:
            unique_in_orig, inverse_indices = _unique_dim_unique_indices_and_inverse(
                sorted_indices,
                is_first,
            )
        else:
            unique_in_orig = _unique_dim_unique_indices(sorted_indices, is_first)

        if return_counts:
            counts = _unique_dim_counts(is_first, size_dim)

    output = _unique_dim_gather_output(moved, unique_in_orig, dim, input.shape)

    return output, inverse_indices, counts
