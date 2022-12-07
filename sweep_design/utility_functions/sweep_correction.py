from ..signal import Signal
from .emd_analyze import get_IMFs_emd
from .a_t import tukey_a_t


def correct_sweep(signal: Signal, start_window: float = None) -> Signal:
    '''Apply correction to sweep signal.

    Using the EMD to subtract the last IMF from the displacement and
    if window is not None then apply a window in the star so that the
    displacement starts at zero.

    Args:
        signal (Relation): The sweep signal on which the correction will be
            applied.

        start_window (float, optional): The time interval at the beginning to
            reduce the deviation from zero. If None, then window is not applied.
            Defaults to None.

    Returns:
        Relation: corrected sweep signal.
    '''

    displacement = signal.integrate().integrate()
    x = displacement.x.array

    IMFs = get_IMFs_emd(displacement)
    if start_window is not None:
        window = tukey_a_t(x, start_window, "left")
        new_displacement = IMFs[0] * Signal(x, window)
    else:
        new_displacement = IMFs[0]

    signal = new_displacement.diff().diff()

    return signal
