from typing import Any, Callable, Tuple, Union

import numpy as np
from packaging import version

RealNumber = Union[float, int]
Number = Union[float, int, complex]

Frequency = np.ndarray
Time = np.ndarray
ImageSpectrogram = np.ndarray
Envelope = np.ndarray

Spectrogram = Tuple[Time, Frequency, ImageSpectrogram]

Theta = "relation.Relation"

Ftat = Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]


def get_type():
    if version.parse(np.__version__) > version.parse("1.19"):
        from numpy.typing import ArrayLike as NpArrayLike
        return NpArrayLike
    else:
        return Any


ArrayLike = get_type()
