from typing import Union
import math
import random

import numpy as np
from scipy.signal import max_len_seq  # type: ignore

from ..relation import Relation
from ..axis import ArrayAxis, get_array_axis_from_array


def get_m_sequence_code(length_code: int, is_full=False) -> np.ndarray:
    '''Create m-sequence array

    Args:
        length_code (int): length of m-sequence array.

        is_full (bool, optional): parameter to return full sequence if True.
            sequence can be longer then requested length code. If False return
            sequence which size equal length_code.
            Defaults to False.

    Returns:
        np.ndarray: array of m-sequence.
    '''
    len_sequence = math.ceil(math.log((length_code - 1), 2))
    start_sequence = np.array([random.randint(0, 1)
                              for _ in range(len_sequence)])

    if np.all(start_sequence == 0):
        start_sequence[0] = 1

    m_sequence = max_len_seq(nbits=len_sequence, state=start_sequence)[0]

    if is_full:
        return m_sequence

    return m_sequence[:length_code - 1]


def get_relation_m_sequence(time: Union[ArrayAxis, np.ndarray],
                            start_sequence: np.ndarray = None,
                            is_full=False) -> Relation:
    '''Create m-sequence relation.

    Args:
        time (Union[ArrayAxis, np.ndarray]): time of sequence.

        start_sequence (np.ndarray, optional): start array sequence of -1 and 1
            to create m-sequence. Defaults to None.

        is_full (bool, optional): if True return full m-sequence with new
            time with equal size to m-sequence array. Defaults to False.

    Returns:
        Relation: relation of m-sequence.
    '''
    if not isinstance(time, ArrayAxis):
        time = get_array_axis_from_array(time)

    len_seq = math.ceil(math.log((time.size - 1), 2))

    if start_sequence is None:
        start_sequence = np.array([random.randint(0, 1)
                                  for _ in range(len_seq)])

    if np.all(start_sequence == 0):
        start_sequence[0] = 1

    m_sequence = max_len_seq(
        nbits=start_sequence.size,
        state=start_sequence)[0]

    if is_full:
        new_time = ArrayAxis(
            0, time.sample * (m_sequence.size - 1), time.sample)
        return Relation(new_time, m_sequence)

    return Relation(time, m_sequence[:time.size])
