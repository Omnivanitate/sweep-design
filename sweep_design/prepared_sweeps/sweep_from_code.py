from typing import List, Union

import numpy as np

from sweep_design.exc import BadInputError

from ..help_types import Literal
from ..axis import ArrayAxis
from ..relation import Relation
from ..sweep import Sweep


def get_convolution_sweep_and_code(
    code: Union[Relation, List[Literal[-1, 0, 1]]],
    base_sweep: Relation
) -> Sweep:
    '''Get sweep use convolution between sweep and desired code.

    Code can be m-sequence, code zinger and etc.

    Args:
        code (Union[Relation, List[Literal[-1, 0, 1]]): some code.
            Relation or List of -1, 0, 1

        base_sweep (Relation): some base sweep.

    Returns:
        Sweep: result signal use as sweep signal.
    '''
    if not isinstance(code, Relation):
        time_code = ArrayAxis(
            0,
            (len(code) - 1) *
            base_sweep.sample,
            base_sweep.sample)
        code = Relation(time_code, code)

    result = Sweep.convolve(
        code,
        base_sweep)[None:code.size *
                    base_sweep.sample]  # type: ignore

    if isinstance(result, Sweep):
        return result.shift(
            base_sweep.end)

    raise BadInputError('Not enough data!')


def get_code_sweep_segments(
    code: Union[Relation, np.ndarray, List[Literal[-1, 0, 1]]],
    base_sweep: Relation
) -> Sweep:
    '''Get sweep use code to compose a base_sweep.

    Code can be m-sequence, code zinger and etc.
    Args:
        code (Union[Relation, List[Literal[-1, 0, 1]]): some code.
        sweep (Relation): some base sweep.
    '''
    if isinstance(code, Relation):
        code = code.y

    new_sweep = base_sweep * code[0]

    for cnt, v in enumerate(code[1:], 1):

        new_sweep = new_sweep + base_sweep.shift(
            cnt * base_sweep.end + base_sweep.sample
        ) * v

    return Sweep(new_sweep)
