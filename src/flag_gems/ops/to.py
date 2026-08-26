# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import warnings
from typing import Optional

import torch
import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)
_COMPOSITE_IMPLICIT_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeImplicitAutograd
)

# Check if float8_e8m0fnu dtype is available in current PyTorch version
_FLOAT8_E8M0FNU = getattr(torch, "float8_e8m0fnu", None)
_MAX_TRITON_NUMEL = 2**31 - 1


@pointwise_dynamic(
    is_tensor=[
        True,
    ],
    promotion_methods=[(0, "DEFAULT")],
)
@triton.jit
def _to_copy_func(x):
    return x


def _resolve_dtype(x: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is None:
        return x.dtype
    if isinstance(dtype, torch.dtype):
        return dtype
    raise TypeError(f"Unsupported dtype argument type: {type(dtype)!r}")


def _resolve_device(x: torch.Tensor, device: Optional[torch.device]) -> torch.device:
    if device is None:
        return x.device
    return torch.device(device)


def _normalize_memory_format(
    memory_format: Optional[torch.memory_format],
) -> torch.memory_format:
    if memory_format is None:
        return torch.preserve_format
    return memory_format


def _allocate_preserve_format(x: torch.Tensor, empty_kwargs: dict) -> torch.Tensor:
    """Recreate a non-overlapping dense tensor with its original strides."""
    return torch.empty_strided(x.size(), x.stride(), **empty_kwargs)


def _allocate_memory_format(
    x: torch.Tensor,
    memory_format: torch.memory_format,
    empty_kwargs: dict,
) -> torch.Tensor:
    """Allocate with canonical strides for an explicit memory format."""
    if memory_format is torch.preserve_format:
        return _allocate_preserve_format(x, empty_kwargs)

    meta = torch.empty(x.size(), device="meta", memory_format=memory_format)
    return torch.empty_strided(x.size(), meta.stride(), **empty_kwargs)


def _fallback_to_copy(
    x: torch.Tensor,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
) -> torch.Tensor:
    return torch.ops.aten._to_copy.default.redispatch(
        _FALLBACK_KEYSET,
        x,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
        memory_format=memory_format,
    )


def _can_use_triton(
    x: torch.Tensor,
    *,
    target_dtype: torch.dtype,
    target_device: torch.device,
    layout: Optional[torch.layout],
    pin_memory: Optional[bool],
) -> bool:
    if x.layout != torch.strided or (layout is not None and layout != torch.strided):
        return False
    if target_device != x.device or x.device.type == "cpu":
        return False
    if pin_memory is True or x.is_quantized:
        return False
    if x.dtype.is_complex or target_dtype.is_complex:
        return False
    if _FLOAT8_E8M0FNU is not None and (
        x.dtype == _FLOAT8_E8M0FNU or target_dtype == _FLOAT8_E8M0FNU
    ):
        return False
    if x.is_conj() or x.is_neg() or x.has_names():
        return False
    if x.numel() == 0 or x.numel() > _MAX_TRITON_NUMEL:
        return False
    # Gapped and overlapping views require native TensorIterator handling.
    # In particular, some Triton backends cannot safely read a gapped view
    # while writing a tensor with canonical contiguous strides.
    if not torch.ops.aten.is_non_overlapping_and_dense(x):
        return False
    return True


def to_dtype(
    x,
    dtype,
    non_blocking=False,
    copy=False,
    memory_format=None,
):
    """Redispatch an overridden backend kernel to PyTorch's Composite wrapper.

    CANN 8.5's torch-npu registers a PrivateUse1 ``to.dtype`` kernel which does
    not honor ``copy=True``. Registration of this bridge is restricted to that
    backend/version in ``flag_gems.__init__``; the Composite implementation then
    preserves alias semantics and dispatches real copies through ``_to_copy``.
    """
    return torch.ops.aten.to.dtype.redispatch(
        _COMPOSITE_IMPLICIT_KEYSET,
        x,
        dtype,
        non_blocking=non_blocking,
        copy=copy,
        memory_format=memory_format,
    )


# The CANN 8.5 compatibility bridge must replace torch-npu's Python-registered
# PrivateUse1 kernel while a ``use_gems`` context is active.
to_dtype._flag_gems_allow_override = True


# func: _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None,
#   bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
def to_copy(
    x,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
):
    target_dtype = _resolve_dtype(x, dtype)
    target_device = _resolve_device(x, device)
    target_memory_format = _normalize_memory_format(memory_format)

    if not _can_use_triton(
        x,
        target_dtype=target_dtype,
        target_device=target_device,
        layout=layout,
        pin_memory=pin_memory,
    ):
        # Native MUSA currently performs the conversion but omits PyTorch's
        # complex-to-real warning. Keep allocation/copy in the runtime while
        # restoring the public semantic contract here.
        if (
            x.device.type == "musa"
            and x.dtype.is_complex
            and not target_dtype.is_complex
        ):
            warnings.warn(
                "Casting complex values to real discards the imaginary part",
                UserWarning,
                stacklevel=2,
            )
        return _fallback_to_copy(
            x,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=memory_format,
        )

    logger.debug("GEMS TO_COPY")
    empty_kwargs = {"dtype": target_dtype, "device": target_device}
    out = _allocate_memory_format(x, target_memory_format, empty_kwargs)

    return _to_copy_func(x, out0=out)
