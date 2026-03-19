import os
import sys

import pytest
import torch

import flag_gems
from tests.accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


@pytest.mark.select_backward
@pytest.mark.parametrize(
    "shape",
    [
        (10,),
        (4, 8),
        (4, 8, 16),
        (2, 3, 4, 5),
        (8, 16, 32),
        (3, 7, 11),
        (2, 1, 4),
        (64, 512),
        (32, 256, 256),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_accuracy_select_backward(shape, dtype, dim):
    device = flag_gems.device

    ndim = len(shape)
    actual_dim = dim + ndim if dim < 0 else dim

    if actual_dim >= ndim:
        pytest.skip(f"dim {dim} out of range for shape {shape}")

    dim_size = shape[actual_dim]

    indices_to_test = [0, dim_size // 2]
    if dim_size > 1:
        indices_to_test.append(dim_size - 1)

    for index in indices_to_test:
        grad_shape = list(shape)
        grad_shape.pop(actual_dim)

        grad = torch.randn(
            grad_shape,
            device=device,
            dtype=dtype,
        )

        ref = to_reference(
            torch.ops.aten.select_backward(
                grad,
                shape,
                actual_dim,
                index,
            )
        )

        with flag_gems.use_gems():
            res = torch.ops.aten.select_backward(
                grad,
                shape,
                actual_dim,
                index,
            )

        assert res.shape == tuple(shape)
        assert res.dtype == grad.dtype

        gems_assert_close(res, ref, dtype)


@pytest.mark.select_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_select_backward_non_contiguous(dtype):
    device = flag_gems.device

    base_shape = (8, 16, 32)
    x = torch.randn(base_shape, device=device, dtype=dtype)
    x = x.transpose(0, 1)

    shape = x.shape
    dim = 1
    index = min(5, shape[dim] - 1)

    grad_shape = list(shape)
    grad_shape.pop(dim)

    grad = torch.randn(
        grad_shape,
        device=device,
        dtype=dtype,
    )

    ref = to_reference(
        torch.ops.aten.select_backward(
            grad,
            shape,
            dim,
            index,
        )
    )

    with flag_gems.use_gems():
        res = torch.ops.aten.select_backward(
            grad,
            shape,
            dim,
            index,
        )

    gems_assert_close(res, ref, dtype)


@pytest.mark.select_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_select_backward_small_and_edge(dtype):
    device = flag_gems.device

    shape = (1, 1, 1)
    dim = 0
    index = 0

    grad = torch.randn((1, 1), device=device, dtype=dtype)

    ref = to_reference(
        torch.ops.aten.select_backward(
            grad,
            shape,
            dim,
            index,
        )
    )

    with flag_gems.use_gems():
        res = torch.ops.aten.select_backward(
            grad,
            shape,
            dim,
            index,
        )

    gems_assert_close(res, ref, dtype)
