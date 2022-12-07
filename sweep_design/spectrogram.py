
from typing import NamedTuple

import numpy as np

from .axis import ArrayAxis


class Spectrogram(NamedTuple):
    '''Spectrogram of signal.

    Gathering time, frequency and spectrogram image together.
    '''
    time: ArrayAxis
    frequency: ArrayAxis
    spectrogram: np.ndarray
