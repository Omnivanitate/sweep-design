import math
import random
from typing import Union

import numpy as np

from ..sweep import Sweep
from ..uncalculated_sweep import UncalculatedSweep
from ..utility_functions.a_t import tukey_a_t
from ..axis import ArrayAxis, get_array_axis_from_array


class SegmentSmallerThenDtError(Exception):
    pass


def get_shuffle(
    time: Union[ArrayAxis, np.ndarray],
    f_start=1.0,
    f_end=101.0,
    length_time_segments=0.5,
    round_number_frequency: int = None,
    time_tapper=1.0,
) -> Sweep:
    """Create shuffle sweep signal.

    t_tapper in seconds is used to apply tukey function at the end of dwell sweep signal.
    """
    if not isinstance(time, ArrayAxis):
        time = get_array_axis_from_array(time)

    if time.sample >= length_time_segments:
        raise SegmentSmallerThenDtError(
            f"Length of shuffle time segment \
            (length_time_segments - {length_time_segments}) should be grate \
             then sample rate (dt - {time.sample})"
        )

    n_segments = math.ceil(time.end / length_time_segments)

    f_segment = np.linspace(f_start, f_end, n_segments + 1)

    if round_number_frequency:
        f_segment = np.round(
            f_segment, round_number_frequency
        )

    random.shuffle(f_segment)

    x = np.linspace(
        0.0,
        length_time_segments,
        math.ceil(length_time_segments / time.sample) + 1
    )
    my_cos = np.cos(x * np.pi / length_time_segments - np.pi) / 2 + 1 / 2
    f_t = np.array([], dtype='float')
    for f1, f2 in zip(f_segment[:-1], f_segment[1:]):
        f_t = np.append(f_t, (my_cos * (f2 - f1) + f1)[1:])

    a_t = tukey_a_t(time.array, time_tapper)
    uncalculated_sweep = UncalculatedSweep(time, f_t[:-2], a_t)

    return uncalculated_sweep()
