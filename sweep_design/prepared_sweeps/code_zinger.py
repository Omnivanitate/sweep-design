from typing import List, Literal

from ..relation import Relation


def get_code_zinger(
    first_sequence_code_zinger: List[Literal[-1, 1]] = [-1, -1, -1, 1],
    periods=1
) -> List[Literal[-1, 1]]:
    '''Build sequence of code zinger.

    Args:
        first_sequence_code_zinger (Literal[, optional): First period of code
            zinger. Defaults to [-1, -1, -1, 1].

        periods (int, optional): Periods of code zinger.
            Repeat first_sequence_code_zinger n(periods) times. Defaults to 1.

        Returns:
            np.ndarray: array of code zinger.
    '''
    return first_sequence_code_zinger * periods


def get_code_zinger_relation(start_sequence: Relation, periods=1) -> Relation:
    '''Create relation of code zinger.

    Args:
        start_sequence (Relation): start of code zinger.
            Example: Relation([-1, -1, -1, 1], [0, 1, 2, 3])

        periods (int, optional):  Periods of code zinger.
            Repeat first_sequence_code_zinger n(periods) times. Defaults to 1.

        Returns:
            Relation: relation of code zinger.
    '''
    sequence = start_sequence

    if periods > 1:
        for k in range(periods):
            sequence = sequence + start_sequence.shift(start_sequence.end * k)

    return sequence
