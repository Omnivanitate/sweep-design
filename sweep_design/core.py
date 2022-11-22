from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple, Tuple

import numpy as np

from .axis import ArrayAxis


class RelationProtocol(ABC):

    @property
    @abstractmethod
    def array(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def x(self) -> ArrayAxis:
        pass

    @property
    @abstractmethod
    def y(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        pass


class MathOperation(Enum):
    ADD = "__add__"
    RADD = "__radd__"
    SUB = "__sub__"
    RSUB = "__rsub__"
    MUL = "__mul__"
    RMUL = "__rmul__"
    TRUEDIV = "__truediv__"
    RTRUEDIV = "__rtruediv__"
    POW = "__pow__"
    RPOW = "__rpow__"


class Spectrogram(NamedTuple):
    time: np.ndarray
    frequency: np.ndarray
    spectrogram_matrix: np.ndarray
