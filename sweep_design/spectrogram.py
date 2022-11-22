
from typing import NamedTuple

import numpy as np

from .axis import ArrayAxis


class Spectrogram(NamedTuple):
    time: ArrayAxis
    frequency: ArrayAxis
    spectrogram: np.ndarray
