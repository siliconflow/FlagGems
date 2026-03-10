import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def square_func(x):
    return x * x


def square(A):
    logger.debug("GEMS SQUARE")
    return square_func(A)
