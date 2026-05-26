import logging

import torch

logger = logging.getLogger(__name__)


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


def _lex_argsort_rows_composite(flat: torch.Tensor) -> torch.Tensor:
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
    indices = torch.arange(num_rows, dtype=torch.int64, device=device)
    if num_rows <= 1 or num_cols == 0:
        return indices

    group_id = torch.zeros(num_rows, dtype=torch.int64, device=device)
    key_scale = 1 << key_bits

    for col in range(num_cols):
        keys = _monotonic_int64_column(flat, col).index_select(0, indices)
        # Use ``group_id * scale + keys`` rather than ``(group_id << bits) | keys``.
        # Functionally identical because ``keys`` is in ``[0, scale)`` after the
        # monotonic remap, but the multiply/add path avoids the int64 bitwise
        # kernels that some Ascend/NPU backends do not provide.
        composite = group_id * key_scale + keys
        perm = torch.argsort(composite, stable=True)
        indices = indices.index_select(0, perm)
        composite = composite.index_select(0, perm)
        # When running under FlagGems' op interception, the registered int64
        # tensor-vs-tensor comparison ops (and the bool dtype cast) route
        # through float32 and lose precision around 2**24, silently mapping
        # non-equal composite values to ``False``. ``int64 - int64`` followed
        # by tensor-vs-scalar ``ne 0`` is the path that stays exact.
        diff = ((composite[1:] - composite[:-1]) != 0).to(torch.int64)
        group_id = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=device),
                torch.cumsum(diff, dim=0),
            ]
        )
        # Early termination: every row has a unique lex prefix already.
        if group_id[-1].item() == num_rows - 1:
            break
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
        keys = flat_t[col].index_select(0, indices)
        perm = torch.argsort(keys, stable=True)
        indices = indices.index_select(0, perm)
    return indices


def _lex_argsort_rows(flat: torch.Tensor) -> torch.Tensor:
    """Return indices that sort rows of a 2D tensor lexicographically."""
    composite = _lex_argsort_rows_composite(flat)
    if composite is not None:
        return composite
    return _lex_argsort_rows_cascade(flat)


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

    moved = input.movedim(dim, 0).contiguous()
    flat = moved.reshape(size_dim, -1)
    other_numel = flat.size(1)

    sorted_indices = _lex_argsort_rows(flat)

    if size_dim == 1 or other_numel == 0:
        is_first = torch.zeros(size_dim, dtype=torch.bool, device=device)
        is_first[0] = True
    else:
        sorted_flat = flat.index_select(0, sorted_indices)
        ne_adjacent = (sorted_flat[1:] != sorted_flat[:-1]).any(dim=1)
        is_first = torch.cat(
            [torch.ones(1, dtype=torch.bool, device=device), ne_adjacent]
        )

    unique_in_orig = sorted_indices.masked_select(is_first)
    output = torch.index_select(input, dim, unique_in_orig)

    inverse_indices = torch.empty(0, dtype=torch.int64, device=device)
    counts = torch.empty(0, dtype=torch.int64, device=device)

    if return_inverse:
        inverse_in_sorted = torch.cumsum(is_first.to(torch.int64), dim=0) - 1
        inverse_indices = torch.empty(size_dim, dtype=torch.int64, device=device)
        # scatter_ is preferred over index_copy_ here because some backends
        # (e.g. Ascend) silently mishandle index_copy_ with permuted int64
        # index tensors on 1D destinations.
        inverse_indices.scatter_(0, sorted_indices.to(torch.int64), inverse_in_sorted)

    if return_counts:
        first_positions = torch.nonzero(is_first, as_tuple=False).flatten()
        end_positions = torch.cat(
            [
                first_positions[1:],
                torch.tensor([size_dim], dtype=torch.int64, device=device),
            ]
        )
        counts = end_positions - first_positions

    return output, inverse_indices, counts
