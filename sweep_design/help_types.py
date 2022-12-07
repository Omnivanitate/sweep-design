from typing import Any, Callable, Tuple, Union

import numpy as np
from packaging import version

RealNumber = Union[float, int]
'''A real number.'''

Number = Union[float, int, complex]
'''A real or complex number'''

Frequency = np.ndarray
'''The frequency expected by numpy array.'''

Time = np.ndarray
'''The time expected by numpy array.'''

ImageSpectrogram = np.ndarray
'''The image spectrogram expected by numpy 2D array.'''

Envelope = np.ndarray
'''The envelop expected by numpy array.'''

Spectrogram = Tuple[Time, Frequency, ImageSpectrogram]
'''The spectrogram gathering time, frequency and image spectrogram in one
object Tuple.'''


Theta = "relation.Relation"
'''`Relation` of phase.'''

Ftat = Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]
'''The Representation of object from which will be extracted frequency
modulation or amplitude modulation.'''

X = np.ndarray
'''x array'''
Y = np.ndarray
'''y array'''

def get_type():
    '''Get type of array like object from numpy depends on version of numpy.

    Returns:
        _type_: type of array like object.
    '''
    if version.parse(np.__version__) > version.parse("1.19"):
        from numpy.typing import ArrayLike as NpArrayLike
        return NpArrayLike
    else:
        return Any


ArrayLike = get_type()
