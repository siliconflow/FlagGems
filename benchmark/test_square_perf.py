import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES, INT_DTYPES
from benchmark.performance_utils import Benchmark, generate_tensor_input


class SquareBenchmark(Benchmark):
    """
    Benchmark for square operation
    """

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)
            yield inp,

    def set_more_shapes(self):
        more_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        more_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return more_shapes_2d + more_shapes_3d


@pytest.mark.parametrize(
    "op_name, torch_op, dtypes",
    [
        pytest.param(
            "square",
            torch.square,
            FLOAT_DTYPES + INT_DTYPES,
            marks=pytest.mark.square,
        ),
    ],
)
def test_square_benchmark(op_name, torch_op, dtypes):
    bench = SquareBenchmark(
        input_fn=None, op_name=op_name, torch_op=torch_op, dtypes=dtypes
    )
    bench.run()
