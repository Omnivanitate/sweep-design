from typing import Optional
import numpy as np
from scipy.signal.windows import tukey  # type: ignore

from ..help_types import Literal

Location = Literal["left", "right", "both"]


def tukey_a_t(
    time: np.ndarray,
    time_tapper: Optional[float],
    location: Location = "both",
) -> np.ndarray:
    '''Calculate array envelope for signal.

    Args:
        time (np.ndarray): time

        time_tapper (float): time_tapper in time, where coefficient will be equal 1.

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
    if time_tapper is None:
        return np.ones(time.size)

    if time_tapper <= time[int(time.size / 2)]:
        tapper = time[time <= time_tapper].size * 2 / time.size
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
