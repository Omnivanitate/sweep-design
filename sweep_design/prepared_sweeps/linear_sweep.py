import numpy as np

from typing import Union
from ..axis import ArrayAxis
from ..sweep import Sweep
from ..uncalculated_sweep import UncalculatedSweep
from ..utility_functions.f_t import f_t_linear_array
from ..utility_functions.a_t import tukey_a_t


def get_linear_sweep(time: Union[ArrayAxis, np.ndarray], f_start=1.0,
                     f_end=100.0, time_tapper=None) -> Sweep:
    """Create linear sweep.

    time_tapper in seconds is used to apply tukey function to a sweep signal.
    """

    time_array = time if isinstance(time, np.ndarray) else time.array

    f_t = f_t_linear_array(time_array, f_start, f_end)
    a_t = tukey_a_t(time_array, time_tapper)

    unsw = UncalculatedSweep(time, f_t, a_t)
    return unsw()
