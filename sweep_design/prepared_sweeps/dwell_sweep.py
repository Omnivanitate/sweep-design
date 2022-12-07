from typing import Union
import numpy as np

from ..axis import ArrayAxis, get_array_axis_from_array

from ..relation import Relation
from ..sweep import Sweep
from ..uncalculated_sweep import ApriorUncalculatedSweep
from ..utility_functions.ftat_functions import dwell
from ..utility_functions.a_t import tukey_a_t


def get_dwell_sweep(
    time: Union[ArrayAxis, np.ndarray],
    f_start=1.0,
    f_central=5.0,
    f_end=100.0,
    time_tapper=1.0,
    aprior_data: Relation = None,
) -> Sweep:
    """Create a dwell sweep using an a priori data.

    time_tapper in seconds is used to apply tukey function at the end of dwell sweep signal.
    """

    if not isinstance(time, ArrayAxis):
        time = get_array_axis_from_array(time)

    ftat_method = dwell(f_start, f_end, f_central)
    uasw = ApriorUncalculatedSweep(time, aprior_data, ftat_method)
    tukey_array = tukey_a_t(time.array, time_tapper, "both")
    tukey_window = Relation(time, tukey_array)

    return uasw() * tukey_window
