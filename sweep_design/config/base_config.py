from ..defaults import methods as dfm
from ..axis import get_array_axis_from_array


class Config:
    """The configuration file.

    The methods used to calculate the various parameters are provided in the
    configuration file. Also the file contain an other parameters.
    -----------------------------------------------------------------------

    **interpolate_extrapolate_method**:
    > The method by which interpolation and extrapolation are performed.
    > The methods returns a function that takes a new x sequence and
    return a new y sequence.

    > **new_x** = `numpy.ndarray`
    > **new_y** = `numpy.ndarray`

    > Method derived from default function:
    >> sweep_design.math_signals.defaults.methods.interpolate_extrapolate

    > **input**:
    >> **x**: `numpy.ndarray`
    >> **y**: `numpy.ndarray`

    > **output**:
    >> `Callable`[[**new_x**], **new_y**]

    - - -

    **math_operation_method**:
    > The method of basic mathematical operations of addition (+),
    subtraction(-), multiplication (*), division(/), exponentiation (**)
    and their unary operations (+=, -=, *=, /=).

    > Method derived from default function:
    >> `sweep_design.math_signals.defaults.methods.math_operation`

    > **input**:
    >> **x**: `numpy.ndarray`
    >> **y1**: `numpy.ndarray`
    >> **y2**: `Union`[`numpy.ndarray`, `float`, `int`, `complex`]
    >> **name_operation**: `sweep_design.math_signals.defaults.base_structures.MathOperation`

    > **output**:
    >> `Tuple` [
    >> **x**: `numpy.ndarray`,
    >> **y**: `numpy.ndarray`
    >>]

    - - -

    **integrate_one_method**:
    > Method for calculating the integral of a sequence on a segment.
    > Method derived from default function:

    >> `sweep_design.math_signals.defaults.methods.one_integrate`

    > **input**:
    >> **x**: `numpy.ndarray`
    >> **y**: `numpy.ndarray`

    > **output**:
    >> `float`

    - - -

    **integrate_method**:
    > The method by which the integration is performed.
    > Method derived from default function:
    >> `sweep_design.math_signals.defaults.methods.integrate`

    > **input**:
    >> **x**: `numpy.ndarray`
    >> **y**: `numpy.ndarray`
    >> d**x**: float - sequence sample rate x

    > **output**:
    >> Tuple [
        **x**: `numpy.ndarray`,
        **y**: `numpy.ndarray`
    ]

    - - -

    **differentiate_method**:
    > The method by which differentiation is performed.
    > Method derived from default function:
    >> `sweep_design.math_signals.defaults.methods.differentiate`

    > **input**:

    >> **x**: `numpy.ndarray`
    >> **y**: `numpy.ndarray`
    >> **dx**: `float` - sequence sample rate **x**

    > **output**:
    >>Tuple [
        **x**: `numpy.ndarray`,
        **y**: `numpy.ndarray`
    ]

    - - -

    **correlate_method**:
    > The method by which the correlation is performed.
    > Method derived from default function:
    >> `sweep_design.math_signals.defaults.methods.correlate`

    > **input**:
    >> cls: Relation
    >> r1: Relation
    >> r2: Relation

    > **output**:
    >> Tuple [
        **x**: `numpy.ndarray`,
        **y**: `numpy.ndarray`
    ]

    - - -

    **convolve_method**:
    > The method by which the convolution is performed.
    > Method derived from default function:
    >> `sweep_design.math_signals.defaults.methods.convolve`

    > **input**:
    >> cls: Relation
    >> r1: Relation
    >> r2: Relation

    > **output**:
    >> Tuple [
        **x**: `numpy.ndarray`,
        **y**: `numpy.ndarray`
    ]

    - - -

    **get_common_x**:
    > A method by which to find the common sequence of numbers along
    > the x-axis, obtained from two other sequences along the x-axis.
    > Method derived from default function:
    >> `sweep_design.math_signals.defaults.methods.get_common_x`

    > **input**:
    >> **x1**: `numpy.ndarray`
    >> **x2**: `numpy.ndarray`
    >> **dx1**: float  - sample rate first sequence
    >> **dx2**: float  - sample rate second sequence

    > **output**:
    >> **x**: `numpy.ndarray`

    - - -

    **spectrum2signal_method**:
    > Method for converting a spectrum into a signal. (Using Fourier transform)
    > Method derived from default function:
    >> `sweep_design.math_signals.defaults.methods.spectrum2sigmal`

    > **input**:
    >> **frequenc**: `numpy.ndarray`
    **specturm**: `numpy.ndarray`
    **start_time**: `float` = `None`

    > **output**:
    >> Tuple [
        **time**: `numpy.ndarray`,
        **amplitude**: `numpy.ndarray`
    ]

    - - -

    **signal2spectrum_method**:
    > Method for converting a signal into a spectrum. (Using Fourier transform)
    > Method derived from default function:
    >> `sweep_design.math_signals.defaults.methods.spectrum2sigmal`

    > **input**:

    >> **time**: `numpy.ndarray`
    >> **amplitude**: `numpy.ndarray`
    >> **is_start_zero** = `False`

    > **output**:
    >> Tuple [
        **frequency**: `numpy.ndarray`,
        **spectrum**: `numpy.ndarray`
    ]

    - - -

    The above methods can be overridden with your own here, or you can import the
    class `Config` somewhere and override it there.
    (They must be written according to the rules corresponding to
    the **input** and **output** parameters)

    """

    # Methods for the Relation.
    get_array_axis_from_array = get_array_axis_from_array
    interpolate_extrapolate_method = dfm.interpolate_extrapolate
    math_operation = dfm.math_operation
    integrate_one_method = dfm.one_integrate
    integrate_method = dfm.integrate
    integrate_function_method = dfm.integrate_function
    differentiate_method = dfm.differentiate
    correlate_method = dfm.correlate
    convolve_method = dfm.convolve
    get_common_x = dfm.get_common_x

    # Methods for Spectrum and Signal.
    spectrum2signal_method = dfm.spectrum2signal
    signal2spectrum_method = dfm.signal2spectrum
