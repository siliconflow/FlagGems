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

import torch
import triton

from flag_gems.ops.to import (
    _allocate_memory_format,
    _can_use_triton,
    _fallback_to_copy,
    _normalize_memory_format,
    _resolve_device,
    _resolve_dtype,
)
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# Integer/boolean dtypes that produce illegal uitofp/sitofp to bf16/fp16 on MetaX.
# MetaX represents bf16 as i16 in LLVM IR, so uitofp i1/i8/i16/i32/i64 -> bf16
# generates an illegal instruction (uitofp requires a floating-point result type).
_INTEGER_DTYPES = frozenset(
    {torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
)
_PROBLEMATIC_TARGET_DTYPES = frozenset({torch.bfloat16, torch.float16})


@pointwise_dynamic(
    is_tensor=[
        True,
    ],
    promotion_methods=[(0, "DEFAULT")],
)
@triton.jit
def _to_copy_func(x):
    return x


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
        return _fallback_to_copy(
            x,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=memory_format,
        )

    # MetaX-specific: integer/bool -> bf16/fp16 generates illegal uitofp/sitofp
    # because MetaX represents bf16 as i16 in LLVM IR. Fall back to a two-step
    # conversion: int -> float32 -> target_dtype.
    if x.dtype in _INTEGER_DTYPES and target_dtype in _PROBLEMATIC_TARGET_DTYPES:
        logger.debug("GEMS_METAX TO_COPY (int->bf16/fp16 two-step)")
        # Step 1: convert integer to float32 using Triton kernel
        empty_kwargs_f32 = {"dtype": torch.float32, "device": target_device}
        intermediate = _allocate_memory_format(
            x, target_memory_format, empty_kwargs_f32
        )
        _to_copy_func(x, out0=intermediate)

        # Step 2: convert float32 to target bf16/fp16 using Triton kernel
        empty_kwargs = {"dtype": target_dtype, "device": target_device}
        out = _allocate_memory_format(x, target_memory_format, empty_kwargs)
        _to_copy_func(intermediate, out0=out)
        return out

    logger.debug("GEMS_METAX TO_COPY")
    empty_kwargs = {"dtype": target_dtype, "device": target_device}
    out = _allocate_memory_format(x, target_memory_format, empty_kwargs)

    return _to_copy_func(x, out0=out)
