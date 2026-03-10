import pytest
import torch

import flag_gems

from .accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    FLOAT_DTYPES,
    INT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)


@pytest.mark.square
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", ALL_FLOAT_DTYPES + ALL_INT_DTYPES)
def test_accuracy_square(shape, dtype):
    if dtype in ALL_INT_DTYPES:
        inp = torch.randint(low=-100, high=100, size=shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.square(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.square(inp)

    gems_assert_equal(res_out, ref_out)
