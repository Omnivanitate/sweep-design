from ..defaults import sweep_methods as dfsm


class SweepConfig:

    """The configuration file for sweep.

    Several methods are defined for default calculations:

    `spectrogram_method_default`:

    The method by which the spectrogram will be calculated.
    Method derived from default function:
    `sweep_design.defaults.sweep_methods.get_spectrogram`

    Args:
        sweep (relation.Relation): instance of sweep signal.

    Returns:
        Spectrogram: tuple of np.ndarray. The first element is time.
            The second is frequency. The third is matrix M x N of spectrogram.

    ---

    `get_f_t`:

    The method by which frequency versus time will be calculated.
    Method derived from default function:
    `sweep_design.defaults.sweep_methods.get_f_t`

    Args:
        sweep (Relation): instance of sweep signal.

    Returns:
        Relation: instance `Relation`

    ---

    `get_a_t`:

    The method by which the time envelope of the signal will be calculated.
    Method derived from default function:
    `sweep_design.defaults.sweep_methods.get_a_t`

    Args:
        sweep (Relation): instance of sweep signal.

    Returns:
        Relation: instance `Relation`

    ---

    `freq2time`:

    The simple method to extract the time envelope of a sweep signal and
    the time-frequency function to generate a sweep signal from a priori data.
    Method derived from default function:
    `sweep_design.defaults.sweep_methods.simple_freq2time`

    Args:
        spectrum (Spectrum): instance signal of 'Spectrum'

    Returns:
        Tuple[Time, Frequency, Envelope]: simple representation Frequency
            modulation from a prior spectrum.

    ---

    The above methods can be overridden with your own here, or you can import the
    class SweepConfig somewhere and override it there.
    (They must be written according to the rules corresponding to
    the input and output parameters)
    """

    # Methods for Sweep.
    spectrogram_method = dfsm.get_spectrogram
    get_f_t = dfsm.get_f_t
    get_a_t = dfsm.get_a_t

    # Method for ApriorUncalculatedSweep.
    freq2time = dfsm.simple_freq2time
