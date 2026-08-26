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

from typing import Generator

import pytest
import torch

from . import base, consts

_TO_SHAPES = [
    (64,),
    (4096,),
    (1024 * 1024,),
    (4096, 4096),
    (64, 512, 512),
]


def _target_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype != torch.float32 else torch.float16


class ToBenchmark(base.Benchmark):
    DEFAULT_SHAPES = _TO_SHAPES

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(self.DEFAULT_SHAPES)
        self.shape_desc = "(B), M, N"


class ToDtypeBenchmark(ToBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            x = torch.randn(shape, dtype=dtype, device=self.device)
            yield x, _target_dtype(dtype)


@pytest.mark.to_copy
def test_to_dtype():
    benchmark = ToDtypeBenchmark(
        op_name="to_dtype",
        torch_op=torch.ops.aten.to.dtype,
        dtypes=consts.FLOAT_DTYPES,
    )
    benchmark.run()


class ToDeviceBenchmark(ToBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            x = torch.randn(shape, dtype=dtype, device=self.device)
            yield x, self.device, _target_dtype(dtype)


@pytest.mark.to_copy
def test_to_device():
    benchmark = ToDeviceBenchmark(
        op_name="to_device",
        torch_op=torch.ops.aten.to.device,
        dtypes=consts.FLOAT_DTYPES,
    )
    benchmark.run()


class ToOtherBenchmark(ToBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            x = torch.randn(shape, dtype=dtype, device=self.device)
            other = torch.empty((1,), dtype=_target_dtype(dtype), device=self.device)
            yield x, other


@pytest.mark.to_copy
def test_to_other():
    benchmark = ToOtherBenchmark(
        op_name="to_other",
        torch_op=torch.ops.aten.to.other,
        dtypes=consts.FLOAT_DTYPES,
    )
    benchmark.run()


class ToDtypeLayoutBenchmark(ToBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            x = torch.randn(shape, dtype=dtype, device=self.device)
            yield x, {
                "dtype": _target_dtype(dtype),
                "layout": torch.strided,
                "device": x.device,
                "pin_memory": False,
                "memory_format": torch.preserve_format,
            }


@pytest.mark.to_copy
def test_to_dtype_layout():
    benchmark = ToDtypeLayoutBenchmark(
        op_name="to_dtype_layout",
        torch_op=torch.ops.aten.to.dtype_layout,
        dtypes=consts.FLOAT_DTYPES,
    )
    benchmark.run()
