from typing import Callable

import numpy as np


def f_t_linear_function(time_start=0., time_end=10., f_start=1., f_end=100.) \
        -> Callable[[np.ndarray], np.ndarray]:
    '''Create liner function of changes frequency-time.

    Use parameters to calculate linear function f(t) = b*t+k, where t is time,
    f(t) function frequency-time.

    Args:
        time_start (RealNumber, optional): start time. Defaults to 0..
        time_end (RealNumber, optional): end time. Defaults to 10..
        f_start (RealNumber, optional): start frequency. Defaults to 1..
        f_end (RealNumber, optional): end frequency. Defaults to 100..

    Returns:
        Callable[[np.ndarray], np.ndarray]: linear function frequency-time.
    '''
    return lambda t: t * (f_end - f_start) / (time_end - time_start) + f_start


def f_t_linear_array(time: np.ndarray, f_start=1., f_end=100.) -> np.ndarray:
    '''Create array of number describe linear changes frequency-time.

    Use parameters to calculate linear function f(t) = b*t+k, where t is time,
    f(t) function frequency-time.

    Args:
        time (np.ndarray): time changes array.
        f_start (RealNumber, optional): start frequency. Defaults to 1..
        f_end (RealNumber, optional): end frequency. Defaults to 100..

    Returns:
        np.ndarray: array of numbers describe linear changes frequency-time.
    '''
    return time * (f_end - f_start) / (time[-1] - time[0]) + f_start
