from typing import Callable, Optional, TypeVar, Union

import numpy as np

from ..relation import Relation
from ..signal import Signal
from .emd_analyze import get_IMFs_emd
from .a_t import tukey_a_t


def soft_clip(
    data: np.ndarray, limits: float, percent=0.85, coefficient: float = 1
) -> np.ndarray:
    """Custom function for correction amplitude."""
    hard_limit = limits
    linear_limit = limits * percent

    def select_choice(data: np.ndarray):

        amplitude = abs(data[0])

        if amplitude <= linear_limit:
            return np.array((data[0], data[1]))

        if amplitude >= hard_limit:
            return np.array(
                (data[0], ((hard_limit / amplitude) ** coefficient) * data[1]))

        scale = hard_limit - linear_limit
        compression = scale * \
            np.sin(np.pi / 2 * (amplitude - linear_limit) / scale)

        return np.array((data[0], (((linear_limit + compression) / amplitude)
                                   ** coefficient) * data[1]))

    return np.apply_along_axis(select_choice, 1, data)


def get_correction_for_source(
    signal: Signal,
    reaction_mass: float = 1.0,
    limits: float = None,
    limits_percent=0.85,
    limit_iteration: Optional[int] = 10,
    window_percent=0.01,
    coefficient_function: Callable[[float], float] = lambda x: x,
) -> Signal:
    '''Sweep signal correction for realization on the vibration source.

    Steps of corrections:
    1. Calculate displacement from force.
    2. Using EMD find first IMFs of displacement.
    3. Apply suppression amplitude after limits.
    4. Apply window at the start to ensure a zero first amplitude.
    5. Return calculated force from displacement

    Args:
        signal (Signal): signal to be corrected.

        reaction_mass (float, optional): reaction mass of source. Defaults to 1.0.

        limits (float, optional): displacement limitation of source. Defaults to None.

        limits_percent (float, optional): from 0 to 1, determine the limits =
            `limits*limits_percent` up to which the displacement amplitude will
            not changed, after that, the limit amplitude will be changed using the
            `soft_clip` function. Defaults to 0.85.

        limit_iteration (Optional[int], optional): iterate corrections. Defaults to 10.

        window_percent (float, optional): apply window at the initial displacement
            to ensure a zero first amplitude. Defaults to 0.01.

        coefficient_function (_type_, optional): function to suppress..
            Defaults to lambda x:x.

    Returns:
        Signal: correct force signal.
    '''

    new_time = signal.x.copy()
    new_time.start = new_time.start + 2 * new_time.sample

    window = Relation(
        new_time,
        tukey_a_t(
            new_time.array,
            new_time.end *
            window_percent,
            "left"))

    displacement = signal.integrate().integrate() / reaction_mass

    new_displacement = None
    imfs = get_IMFs_emd(displacement)
    imfs[0] = imfs[0] * window

    d_array = np.vstack((imfs[0].y, signal.y[1:-1]))
    d_array = np.transpose(d_array)

    cnt: int = 0
    while True and limits is not None:
        cnt += 1
        result = d_array

        for _ in range(cnt):
            result = soft_clip(
                result,
                limits,
                limits_percent,
                coefficient_function(cnt))

        force = type(signal)(new_time, result[:, 1])
        new_displacement = force.integrate().integrate() / reaction_mass
        imfs = get_IMFs_emd(new_displacement)

        new_displacement = imfs[0] * window

        if np.all(np.abs(new_displacement.y) < limits):
            break

        if limit_iteration is not None and cnt + 1 > limit_iteration:
            break

    new_displacement = new_displacement or imfs[0]

    return new_displacement.diff().diff() * reaction_mass
