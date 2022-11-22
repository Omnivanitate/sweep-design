from copy import copy
from typing import Optional

import numpy as np

from .types import ArrayLike, RealNumber


def get_actual_sample(x: np.ndarray) -> RealNumber:
    '''Calculate actual sample. Use for self-test. Problem with floating point
    in Python (https://docs.python.org/3/tutorial/floatingpoint.html).

    Args:
        x (np.ndarray): array of numbers.

    Returns:
        RealNumber: an actual sample
    '''
    diff = np.min(np.diff(x))
    values, counts = np.unique(diff, return_counts=True)
    common_sample = values[np.argmax(counts)]
    return common_sample


class ArrayAxis:

    def __init__(self, start: RealNumber, end: RealNumber,
                 sample: RealNumber) -> None:
        '''The representation of some array axis.

        This is description of sequence. The start sequence. The end sequence.
        And the sample spacing between elements.

        Args:
            start (RealNumber): is start of array.
            end (RealNumber): is end of array.
            sample (RealNumber): is sample of array.
        '''
        self._start = start
        self._end = end
        self._sample = sample
        self._array: Optional[np.ndarray] = None
        self._actual_sample: Optional[RealNumber] = None

    def _reset_property(self) -> None:
        self._array = None
        self._actual_sample = None

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            self._array = np.linspace(
                self._start, self._end,
                round(abs((self._end - self._start) / self._sample)) + 1
            )

        return self._array

    @property
    def start(self) -> RealNumber:
        return self._start

    @start.setter
    def start(self, value: RealNumber) -> None:
        self._start = value
        self._reset_property()

    @property
    def end(self) -> RealNumber:
        return self._end

    @end.setter
    def end(self, value: RealNumber) -> None:
        self._end = value
        self._reset_property()

    @property
    def actual_sample(self) -> RealNumber:
        if self._actual_sample is None:
            self._actual_sample = get_actual_sample(self.array)
        return self._actual_sample

    @property
    def sample(self) -> RealNumber:
        return self._sample

    @sample.setter
    def sample(self, value: RealNumber) -> None:
        self._sample = value
        self._reset_property()

    @property
    def size(self) -> int:
        return self.array.size

    def copy(self):
        return copy(self)

    def __str__(self):
        result = f"start: {self._start}\n" \
            f"end: {self._end}\n" \
            f"dx: {self._sample}\n" \
            f"calculated dx: {self._actual_sample}"

        return result


def get_array_axis_from_array(
        x: ArrayLike, round_dx: bool = True) -> ArrayAxis:
    '''Create instance of Axis from some array of numbers.

    Args:
        x (ArrayLike): input array_like of numbers.
    '''
    x = np.array(x)
    dx = get_actual_sample(x)
    if round_dx and isinstance(dx, (int, float)):
        dx = 1 / round(1 / dx) if dx < 1 else round(dx)

    return ArrayAxis(start=x[0], end=x[-1], sample=dx)
