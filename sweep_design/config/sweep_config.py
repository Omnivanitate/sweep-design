from ..defaults import sweep_methods as dfsm


class SweepConfig:

    """The configuration file for sweep.

    > Several methods are defined for default calculations:

    **spectrogram_method_default**
    > The method by which the spectrogram will be calculated.
    > Method derived from default function:
    >> `sweep_design.math_signals.defaults.sweep_methods.get_spectrogram`

    > **input**:
    >> **time**: `numpy.ndarray`
    **amplitude**: np.ndarry
    **dt**: float

    > **output**:
    >> `Tuple` [
    >> **time**: `numpy.ndarray`,
    >> **frequency**: `numpy.ndarray`,
    >> **spectrogram**: `numpy.ndarray`
    ]

    - - -

    **get_f_t**
    > The method by which frequency versus time will be calculated.
    > Method derived from default function:
    >> `sweep_design.math_signals.defaults.sweep_methods.get_f_t`

    > **input**:
    >> **time**: `numpy.ndarray`
    **amplitude**: `numpy.ndarray`

    > **output**:
    >> `sweep_design.math_signals.math_relation.Relation`

    - - -

    **get_a_t**
    > The method by which the time envelope of the signal will be calculated.
    > Method derived from default function:
    >> `sweep_design.math_signals.defaults.sweep_methods.get_a_t`

    > **input**:
    >> **time**: `numpy.ndarray`
    **amplitude**: `numpy.ndarray`

    > **output**:
    >> `sweep_design.math_signals.math_relation.Relation`

    - - -

    **integrate_function**
    > A method for integrating a function rather than a sequence.
    Method derived from default function:
    >> `sweep_design.math_signals.defaults.sweep_methods.integrate_quad`

    > **input**:
    >> **f_t_function**: `Callable`[[**time**: `numpy.ndarray`],
    **frequency**: `numpy.ndarray`]
    **time**: `numpy.ndarray`

    > **output**:
    >> **y**: `numpy.ndarray`

    - - -

    **freq2time**
    > The simple method to extract the time envelope of a sweep signal and
    the time-frequency function to generate a sweep signal from a priori data.
    Method derived from default function:
    >> `sweep_design.math_signals.defaults.sweep_methods.simple_freq2time`

    > **input**:
    >> **spectrum**: `sweep_design.math_signals.math_signal.Spectrum`

    > **output**:
    >> Tuple [
    >> **time**: `numpy.ndarray`,
    **frequency**: `numpy.ndarray`,
    **envelope**: `numpy.ndarray`
    ]

    - - -

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
