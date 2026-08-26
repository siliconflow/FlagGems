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

import itertools
import warnings

import pytest
import torch

import flag_gems
from flag_gems.ops.to import to_copy as generic_to_copy

from . import accuracy_utils as utils

_TO_SHAPE = (2, 3, 4, 5)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.COMPLEX_DTYPES,
)
def test_to_dtype(shape, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype in utils.COMPLEX_DTYPES:
        pytest.skip("#2855: Skiping complex to_copy test on tsingmicro platform")
    if flag_gems.vendor_name == "ascend" and dtype in utils.COMPLEX_DTYPES:
        pytest.skip("Issues #3267: Ascend NPU does not support complex32 dtype")
    x = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = ref_x.to(dtype)
    with flag_gems.use_gems():
        out = x.to(dtype)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("target_dtype", utils.ALL_FLOAT_DTYPES + utils.COMPLEX_DTYPES)
def test_to_copy_dtype_cast(shape, target_dtype):
    if flag_gems.vendor_name == "tsingmicro" and target_dtype in utils.COMPLEX_DTYPES:
        pytest.skip("#2855: Skiping complex to_copy test on tsingmicro platform")
    if flag_gems.vendor_name == "ascend" and target_dtype in utils.COMPLEX_DTYPES:
        pytest.skip("Issues #3267: Ascend NPU does not support complex32 dtype")
    src_dtype = torch.float32 if target_dtype != torch.float32 else torch.float16
    x = torch.randn(shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=target_dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=target_dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
def test_to_dtype_overload_alias_copy_and_cast():
    x = torch.randn(_TO_SHAPE, dtype=torch.float32, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    with flag_gems.use_gems():
        alias = torch.ops.aten.to.dtype(x, x.dtype)
        copied = torch.ops.aten.to.dtype(x, x.dtype, copy=True)
        cast = torch.ops.aten.to.dtype(x, torch.float16)

    assert alias is x
    assert alias.data_ptr() == x.data_ptr()
    assert copied is not x
    assert copied.data_ptr() != x.data_ptr()
    utils.gems_assert_equal(copied, ref_x)
    utils.gems_assert_equal(cast, ref_x.to(torch.float16))


@pytest.mark.to_copy
def test_to_dtype_overload_autograd():
    x = torch.randn(
        (8,), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )

    with flag_gems.use_gems():
        out = torch.ops.aten.to.dtype(x, torch.float16)
        out.float().sum().backward()

    utils.gems_assert_equal(x.grad, torch.ones((8,), dtype=x.dtype))


@pytest.mark.to_copy
def test_to_device_overload_alias_copy_and_transfer():
    x = torch.randn(_TO_SHAPE, dtype=torch.float32, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    requested_device = torch.device(flag_gems.device)

    with flag_gems.use_gems():
        alias = torch.ops.aten.to.device(x, requested_device, x.dtype)
        copied = torch.ops.aten.to.device(x, requested_device, x.dtype, copy=True)
        cpu_out = torch.ops.aten.to.device(
            x, torch.device("cpu"), x.dtype, non_blocking=True
        )
        device_out = torch.ops.aten.to.device(
            ref_x, requested_device, ref_x.dtype, non_blocking=True
        )

    assert alias is x
    assert alias.data_ptr() == x.data_ptr()
    assert copied is not x
    assert copied.data_ptr() != x.data_ptr()
    utils.gems_assert_equal(copied, ref_x)
    utils.gems_assert_equal(cpu_out, ref_x)
    utils.gems_assert_equal(device_out, ref_x)


@pytest.mark.to_copy
def test_to_device_overload_cross_accelerator_request():
    if flag_gems.runtime.torch_device_fn.device_count() < 2:
        pytest.skip("Cross-device transfer requires at least two visible devices")

    source_device = torch.device(torch.device(flag_gems.device).type, 0)
    target_device = torch.device(source_device.type, 1)
    x = torch.randn(_TO_SHAPE, dtype=torch.float32, device=source_device)
    ref_x = utils.to_reference(x)

    with flag_gems.use_gems():
        out = torch.ops.aten.to.device(x, target_device, x.dtype, non_blocking=True)

    assert out.device == target_device
    utils.gems_assert_equal(out, ref_x)


@pytest.mark.to_copy
def test_to_other_overload_uses_options_not_shape():
    x = torch.randn(_TO_SHAPE, dtype=torch.float32, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    same_options = torch.empty((1,), dtype=x.dtype, device=x.device)
    other = torch.empty((1,), dtype=torch.float16, device=x.device)

    with flag_gems.use_gems():
        alias = torch.ops.aten.to.other(x, same_options)
        out = torch.ops.aten.to.other(x, other)
        copied = torch.ops.aten.to.other(x, same_options, copy=True)

    assert alias is x
    assert out.shape == x.shape
    assert out.dtype == other.dtype
    assert out.device == other.device
    assert out.layout == other.layout
    assert copied.data_ptr() != x.data_ptr()
    utils.gems_assert_equal(out, ref_x.to(other.dtype))
    utils.gems_assert_equal(copied, ref_x)


@pytest.mark.to_copy
def test_to_dtype_layout_overload_alias_copy_and_memory_format():
    x = torch.randn(_TO_SHAPE, dtype=torch.float32, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    with flag_gems.use_gems():
        alias = torch.ops.aten.to.dtype_layout(
            x,
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
            pin_memory=False,
            memory_format=torch.contiguous_format,
        )
        copied = torch.ops.aten.to.dtype_layout(
            x,
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
            pin_memory=False,
            copy=True,
            memory_format=torch.preserve_format,
        )
        channels_last = torch.ops.aten.to.dtype_layout(
            x,
            dtype=torch.float16,
            layout=torch.strided,
            device=x.device,
            pin_memory=False,
            memory_format=torch.channels_last,
        )

    assert alias is x
    assert copied.data_ptr() != x.data_ptr()
    assert channels_last.is_contiguous(memory_format=torch.channels_last)
    utils.gems_assert_equal(copied, ref_x)
    utils.gems_assert_equal(channels_last, ref_x.to(torch.float16))


@pytest.mark.to_copy
@pytest.mark.parametrize(
    "memory_format",
    [torch.preserve_format, torch.contiguous_format],
)
def test_to_copy_preserve_strides(memory_format):
    base = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    x = base.transpose(0, 1)[::2]
    ref_base = utils.to_reference(base)
    ref_x = ref_base.transpose(0, 1)[::2]
    ref_out = torch.ops.aten._to_copy(
        ref_x,
        dtype=ref_x.dtype,
        memory_format=memory_format,
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(
            x,
            dtype=x.dtype,
            memory_format=memory_format,
        )
    utils.gems_assert_equal(res_out, ref_out)
    if memory_format is torch.preserve_format:
        assert res_out.stride() == ref_out.stride()
    else:
        assert res_out.is_contiguous()


@pytest.mark.to_copy
def test_to_copy_same_options_always_copies_and_accepts_pin_memory_false():
    x = torch.randn((4, 8), dtype=torch.float32, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    with flag_gems.use_gems():
        out = torch.ops.aten._to_copy(
            x,
            dtype=x.dtype,
            device=x.device,
            pin_memory=False,
        )

    assert out is not x
    assert out.data_ptr() != x.data_ptr()
    utils.gems_assert_equal(out, ref_x)


@pytest.mark.to_copy
def test_to_copy_preserve_non_overlapping_dense_strides():
    base = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    x = base.transpose(0, 1)
    ref_base = utils.to_reference(base)
    ref_x = ref_base.transpose(0, 1)
    assert torch.ops.aten.is_non_overlapping_and_dense(x)

    ref_out = torch.ops.aten._to_copy(
        ref_x, dtype=torch.float16, memory_format=torch.preserve_format
    )
    with flag_gems.use_gems():
        out = torch.ops.aten._to_copy(
            x, dtype=torch.float16, memory_format=torch.preserve_format
        )

    utils.gems_assert_equal(out, ref_out)
    assert out.stride() == ref_out.stride() == x.stride()


@pytest.mark.to_copy
@pytest.mark.parametrize(
    "shape,memory_format",
    [
        ((2, 3, 4, 5), torch.channels_last),
        ((2, 3, 4, 5, 6), torch.channels_last_3d),
    ],
)
def test_to_copy_explicit_memory_format(shape, memory_format):
    x = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(
        ref_x, dtype=torch.float16, memory_format=memory_format
    )

    with flag_gems.use_gems():
        out = torch.ops.aten._to_copy(
            x, dtype=torch.float16, memory_format=memory_format
        )

    utils.gems_assert_equal(out, ref_out)
    assert out.stride() == ref_out.stride()
    assert out.is_contiguous(memory_format=memory_format)


@pytest.mark.to_copy
def test_to_copy_overlapping_preserve_format_fallback():
    base = torch.randn((1, 8), dtype=torch.float32, device=flag_gems.device)
    x = base.expand(4, 8)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(
        ref_x, dtype=torch.float16, memory_format=torch.preserve_format
    )

    with flag_gems.use_gems():
        out = torch.ops.aten._to_copy(
            x, dtype=torch.float16, memory_format=torch.preserve_format
        )

    utils.gems_assert_equal(out, ref_out)
    assert out.stride() == ref_out.stride()


@pytest.mark.to_copy
def test_to_copy_layout_mismatch_uses_native_error():
    x = torch.randn((4, 8), dtype=torch.float32)

    with pytest.raises(RuntimeError):
        generic_to_copy(x, layout=torch.sparse_coo)


@pytest.mark.to_copy
def test_to_copy_sparse_cpu_fallback():
    indices = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.int64)
    values = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
    x = torch.sparse_coo_tensor(indices, values, (2, 3)).coalesce()
    ref_out = torch.ops.aten._to_copy(x, dtype=torch.float64)
    out = generic_to_copy(x, dtype=torch.float64)

    assert out.layout == ref_out.layout
    assert out.is_coalesced() == ref_out.is_coalesced()
    torch.testing.assert_close(out.indices(), ref_out.indices())
    torch.testing.assert_close(out.values(), ref_out.values())


@pytest.mark.to_copy
def test_to_copy_quantized_cpu_fallback():
    try:
        x = torch.quantize_per_tensor(
            torch.randn((4, 8)), scale=0.1, zero_point=3, dtype=torch.quint8
        )
    except (NotImplementedError, RuntimeError) as exc:
        pytest.skip(f"Native CPU quantization is unavailable: {exc}")

    ref_out = torch.ops.aten._to_copy(x)
    out = generic_to_copy(x)

    assert out.qscheme() == ref_out.qscheme()
    assert out.q_scale() == ref_out.q_scale()
    assert out.q_zero_point() == ref_out.q_zero_point()
    torch.testing.assert_close(out.int_repr(), ref_out.int_repr())


@pytest.mark.to_copy
@pytest.mark.skipif(
    flag_gems.vendor_name in ("ascend", "tsingmicro"),
    reason="The target runtime does not support complex tensors",
)
def test_to_copy_complex_to_real_warns(capfd):
    x = torch.randn((4, 8), dtype=torch.complex64, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = ref_x.real.to(torch.float32)
    with flag_gems.use_gems():
        warn_always = torch._C._get_warnAlways()
        torch._C._set_warnAlways(True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = torch.ops.aten._to_copy(x, dtype=torch.float32)
        finally:
            torch._C._set_warnAlways(warn_always)

    captured = capfd.readouterr()
    python_warnings = "\n".join(str(item.message) for item in caught)
    assert "Casting complex values to real" in captured.err + python_warnings
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.to_copy
def test_to_copy_lazy_neg_fallback():
    base = torch.arange(8, dtype=torch.float32, device=flag_gems.device)
    x = torch.ops.aten._neg_view.default(base)
    if not x.is_neg():
        pytest.skip("The target runtime resolves lazy negative views eagerly")

    ref_out = torch.ops.aten._to_copy(x)
    with flag_gems.use_gems():
        out = torch.ops.aten._to_copy(x)

    utils.gems_assert_equal(out, utils.to_reference(ref_out))
    assert out.is_neg() == ref_out.is_neg()


# Generate (src, dst) pairs excluding same-dtype conversions
_FLOAT_TO_FLOAT_PAIRS = [
    (s, d)
    for s, d in itertools.product(utils.FLOAT_DTYPES, utils.ALL_FLOAT_DTYPES)
    if s != d
]


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("src_dtype,dst_dtype", _FLOAT_TO_FLOAT_PAIRS)
def test_to_copy_float_to_float(shape, src_dtype, dst_dtype):
    if flag_gems.vendor_name == "ascend" and (
        src_dtype == torch.bfloat16 or dst_dtype == torch.bfloat16
    ):
        pytest.skip("Ascend NPU may have issues with bfloat16")
    x = torch.randn(shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("src_dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("dst_dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_to_copy_float_to_int(shape, src_dtype, dst_dtype):
    if flag_gems.vendor_name == "ascend" and src_dtype == torch.bfloat16:
        pytest.skip("Ascend NPU may have issues with bfloat16")
    x = torch.randn(shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("src_dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.parametrize("dst_dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_to_copy_int_to_float(shape, src_dtype, dst_dtype):
    if flag_gems.vendor_name == "ascend" and dst_dtype == torch.bfloat16:
        pytest.skip("Ascend NPU may have issues with bfloat16")
    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support int8/int16 generation.
        x = torch.randint(-100, 100, shape, dtype=src_dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        x = torch.randint(-100, 100, shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)


# Generate (src, dst) int pairs excluding same-dtype conversions
_INT_DTYPES = [torch.int8, torch.int16, torch.int32]
_INT_TO_INT_PAIRS = list(itertools.permutations(_INT_DTYPES, 2))


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("src_dtype,dst_dtype", _INT_TO_INT_PAIRS)
def test_to_copy_int_to_int(shape, src_dtype, dst_dtype):
    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support int8/int16 generation.
        x = torch.randint(-100, 100, shape, dtype=src_dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        x = torch.randint(-100, 100, shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("src_dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_to_copy_float_to_uint8(shape, src_dtype):
    if flag_gems.vendor_name == "ascend" and src_dtype == torch.bfloat16:
        pytest.skip("Ascend NPU may have issues with bfloat16")
    x = torch.randint(0, 255, shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=torch.uint8)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=torch.uint8)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dst_dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_to_copy_uint8_to_float(shape, dst_dtype):
    if flag_gems.vendor_name == "ascend" and dst_dtype == torch.bfloat16:
        pytest.skip("Ascend NPU may have issues with bfloat16")
    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support int8/int16 generation.
        x = torch.randint(0, 255, shape, dtype=torch.uint8, device="cpu").to(
            flag_gems.device
        )
    else:
        x = torch.randint(0, 255, shape, dtype=torch.uint8, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dst_dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_to_copy_uint8_to_int(shape, dst_dtype):
    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support int8/int16 generation.
        x = torch.randint(0, 255, shape, dtype=torch.uint8, device="cpu").to(
            flag_gems.device
        )
    else:
        x = torch.randint(0, 255, shape, dtype=torch.uint8, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)
