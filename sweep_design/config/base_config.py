from ..defaults import methods as dfm
from ..axis import get_array_axis_from_array


class Config:
    """The configuration class.

    The methods used to calculate the various parameters are provided in the
    configuration file. Also the file contain an other parameters.

    ---

    `get_array_axis_from_array_method`:

    Method to create new instance of ArrayAxis from some array of numbers.

    Args:
        x (ArrayLike): input array_like of numbers.
        round_dx (bool, optional): if True then round sample. Defaults to True.

    Returns:
        ArrayAxis: new ArrayAxis.

    ---


    `interpolate_extrapolate_method`:

    The method by which interpolation and extrapolation are performed.
    The method returns a function that takes a new x sequence and
    return a new y sequence.

    Method derived from default function:
    sweep_design.defaults.methods.interpolate_extrapolate

    Args:
        x (X): numbers array of axis. Samples can be not equal.

        y (Y): Representation interpolated extrapolated functions
            as array.

        bounds_error (bool, optional): if False then do not raise error if new
            array behind of bound old array. Defaults to False.

        fill_value (float, optional): default fill value if other not expected.
            Defaults to 0.0.

    Returns:
        Callable[[X], Y]: Callable that get first new array of x and return
            interpolate-extrapolate result.

    ---

    `math_operation_method`:

    Method for math operation
    Method derived from default function:
    `sweep_design.defaults.math_operation`

    Args:
        y1 (np.ndarray): first sequence y.
        y2 (Union[np.ndarray, Number]): second sequence y or other number
            name_operation (MathOperation): which mathematical operation (+, -,
            \\*, / and etc.)

    Raises:
        TypeFuncError: if operation can not be executed.

    Returns:
        Y: result of math operation.

    ---

    `integrate_one_method`:

    Method for calculating the integral of a sequence on a segment.
    Method derived from default function:
    `sweep_design.defaults.methods.one_integrate`

    Args:
        relation (Relation): from will be calculated integral.

    Returns:
        float: result of integration.

    ---

    `integrate_method`:

    The method by which the integration array is performed. Integration across
    the entire function. Get the expected integrated array function.
    Method derived from default function:
    `sweep_design.defaults.methods.integrate`

    Args:
        relation (Relation): integrated function.

    Returns:
        Tuple[XAxis, Y]: result of integration of function.

    ---

    `integrate_function_method`:

    The method by which the integration function is performed. Integration across
    the entire function. Get the expected integrated array function.
    Method derived from default function:
    `sweep_design.defaults.methods.integrate_function`

    Args:
        function (Callable[[x], y]): function is describing
            changes frequency from time.

        x (np.ndarray): time array.

    Returns:
        Relation: result of integration function.

    ---

    `differentiate_method`:

    The method by which differentiation is performed.
    Method derived from default function:
    `sweep_design.defaults.methods.differentiate`

    Args:
        relation (Relation): function which will be differentiated.

    Returns:
        Tuple[XAxis, Y]: result of differentiation.

    ---

    `correlate_method`:

    The method by which the correlation is performed.
    Method derived from default function:
    `sweep_design.defaults.methods.correlate`

    Args:
        cls (Type[&quot;Relation&quot;]): cls to use equalization of two arrays.
        r1 (Relation): first function y.
        r2 (Relation): second function y.

    Returns:
        Tuple[XAxis, np.ndarray]: result of correlation.

    ---

    `convolve_method`:

    The method by which the convolution is performed.
    Method derived from default function:
    `sweep_design.defaults.methods.convolve`

    Args:
        cls (Type[&quot;Relation&quot;]): class to use equalization of two arrays.
        r1 (Relation): first function y.
        r2 (Relation): second function y.

    Returns:
        Tuple[XAxis, np.ndarray]: result of convolution.

    ---

    `get_common_x`:

    A method by which to find the common sequence of numbers along
    the x-axis, obtained from two other sequences along the x-axis.
    Method derived from default function:
    `sweep_design.defaults.methods.get_common_x`

    Args:
        x1 (XAxis): first axis.
        x2 (XAxis): second axis.

    Returns:
        XAxis: return common axis.

    ---

    `spectrum2signal_method`:

    Method for converting a spectrum into a signal. (Using Fourier transform)
    Method derived from default function:
    `sweep_design.defaults.methods.spectrum2signal`

    Args:
        relation (Relation): spectrum of signal.
        time_start (float, optional): default fft convert to 0. time. Maybe you
            want another start of time. Defaults to None.

    Returns:
        Tuple[TimeAxis, np.ndarray]: result transformation spectrogram to signal.

    ---

    `signal2spectrum_method`:
    Method for converting a signal into a spectrum. (Using Fourier transform)
    Method derived from default function:
    `sweep_design.defaults.methods.signal2spectrum`

    Args:
        relation (Relation): signal from which get spectrum.
        is_start_zero (bool, optional): Consider array started from zero time.
            Defaults to False.

    Returns:
        Tuple[FrequencyAxis, np.ndarray]: result transformation signal to
            spectrum.

    ---

    The above methods can be overridden with your own here, or you can import the
    class `Config` somewhere and override it there.
    (They must be written according to the rules corresponding to
    the **input** and **output** parameters)

    """

    # Methods for the Relation.
    get_array_axis_from_array_method = get_array_axis_from_array
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
