from typing import Literal
import numpy as np
from scipy.signal.windows import tukey  # type: ignore


def tukey_a_t(
    time: np.ndarray,
    t_tapper: float,
    location: Literal["left", "right", "both"] = "both",
) -> np.ndarray:
    '''Calculate array envelope for signal.

    Args:
        time (np.ndarray): time

        t_tapper (float): t_tapper in time, where coefficient will be equal 1.

        location (Literal[&quot;left&quot;, &quot;right&quot;, &quot;both&quot;], optional):
        Where the correction will be applied.
        "left" is at the start.
        "right" is at the end.
        "both" is at the start and at the end.
        Defaults to "both".

    Returns:
        np.ndarray: amplitude correction for signal. Multiple signal to result
        of function.
    '''

    if t_tapper <= time[int(time.size / 2)]:
        tapper = time[time <= t_tapper].size * 2 / time.size
    else:
        tapper = 1.0

    result = tukey(time.size, alpha=tapper)

    if location == "both":
        return result

    result = np.append(
        result[: int(time.size / 2)], np.ones(time.size - int(time.size / 2))
    )

    if location == "left":
        return result

    if location == "right":
        return np.flip(result)

    return np.ones(time.size)
