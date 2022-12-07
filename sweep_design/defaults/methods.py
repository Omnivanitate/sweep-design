"""This is where default methods are defined."""
from typing import TYPE_CHECKING, Callable, Tuple, Type, Union

import numpy as np
import scipy  # type: ignore
from packaging import version
from scipy.interpolate import interp1d  # type: ignore

from sweep_design import exc  # type: ignore

from ..axis import ArrayAxis, get_array_axis_from_array
from ..help_types import X, Y
from ..core import MathOperation
from ..help_types import Number
from ..exc import TypeFuncError

if version.parse(scipy.__version__) < version.parse("1.6.0"):
    from scipy.integrate import cumtrapz, quad, trapz  # type: ignore
    integration = trapz
    cumulative_integration = cumtrapz
    quad_integrate_function = quad
else:
    from scipy.integrate import (cumulative_trapezoid, quad,  # type: ignore
                                 trapezoid)
    integration = trapezoid
    cumulative_integration = cumulative_trapezoid
    quad_integrate_function = quad

XAxis = ArrayAxis
'''Array axis of `x`.'''

FrequencyAxis = ArrayAxis
'''Array axis of frequency.'''

TimeAxis = ArrayAxis
'''Array axis of time.'''

if TYPE_CHECKING:
    from ..relation import Relation


def math_operation(
    y1: np.ndarray,
    y2: Union[np.ndarray, Number],
    name_operation: MathOperation,
) -> Y:
    '''Math operations.

    Using numpy math operations.

    Args:
        y1 (np.ndarray): first sequence y.
        y2 (Union[np.ndarray, Number]): second sequence y or other number
            name_operation (MathOperation): which mathematical operation (+, -,
            \\*, / and etc.)

    Raises:
        TypeFuncError: if operation can not be executed.

    Returns:
        Y: result of math operation.
    '''
    try:
        y = y1.__getattribute__(name_operation.value)(y2)

    except Exception as e:
        raise TypeFuncError(
            name_operation.value.strip("_"), type(y1), type(y2)) from e

    return y


def one_integrate(relation: 'Relation') -> float:
    '''Integration.

    Taking the integral on a segment. Return of the area under the graph.
    using scipy trapezoid integration.

    Args:
        relation (Relation): from will be calculated integral.

    Returns:
        float: result of integration.
    '''
    x, y = relation.get_data()
    return integration(y, x)


def integrate(relation: 'Relation') -> Tuple[XAxis, Y]:
    '''Integration.

    Integration across the entire function. Get the expected integrated
    array function.
    Using the `scipy.integrate.cumtrapz` function.

    Args:
        relation (Relation): integrated function.

    Returns:
        Tuple[XAxis, Y]: result of integration of function.
    '''
    array_axis = relation.x.copy()
    dx = array_axis.sample
    array_axis.start = array_axis.start + array_axis.sample
    return array_axis, cumulative_integration(relation.y) * (dx)


def differentiate(relation: 'Relation') -> Tuple[XAxis, Y]:
    '''Differentiation.

    The method by which differentiation is performed.
    Using the `numpy.diff` function.

    Args:
        relation (Relation): function which will be differentiated.

    Returns:
        Tuple[XAxis, Y]: result of differentiation.
    '''
    array_axis = relation.x.copy()
    dx = array_axis.sample
    array_axis.start = array_axis.start + array_axis.sample / 2
    array_axis.end = array_axis.end - array_axis.sample / 2
    return array_axis, np.diff(relation.y) / (dx)


def interpolate_extrapolate(
    x: X, y: Y, bounds_error=False, fill_value=0.0
) -> Callable[[XAxis], Y]:
    '''Interpolation and extrapolation

    Using the `scipy.interpolate.interp1d` function.
    Returning function of interpolation.

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
    '''
    interpolate_extrapolate = interp1d(x, y, bounds_error=bounds_error,
                                       fill_value=fill_value)

    def wrapper(new_x: XAxis) -> Y:
        new_y = interpolate_extrapolate(new_x.array)
        return new_y

    return wrapper


def get_common_x(x1: XAxis, x2: XAxis) -> XAxis:
    '''Specifies the overall x-axis.

    Finds the general sample rate and beginning and end of sequence.
    A method by which to find the common sequence of numbers along
    the x-axis, obtained from two other sequences along the x-axis.

    Args:
        x1 (XAxis): first axis.
        x2 (XAxis): second axis.

    Returns:
        XAxis: return common axis.
    '''
    dx1 = x1.sample
    dx2 = x2.sample

    dx = dx1 if dx1 <= dx2 else dx2
    x_start = x1.start if x1.start <= x2.start else x2.start
    x_end = x1.end if x1.end >= x2.end else x2.end
    return ArrayAxis(start=x_start, end=x_end, sample=dx)


def correlate(cls: Type["Relation"], r1: "Relation",
              r2: "Relation") -> Tuple[XAxis, np.ndarray]:
    '''Correlation.

    The method by which the correlation is performed.
    Using the `numpy.correlate` function.

    Args:
        cls (Type[&quot;Relation&quot;]): class to use equalization of two arrays.
        r1 (Relation): first function y.
        r2 (Relation): second function y.

    Returns:
        Tuple[XAxis, np.ndarray]: result of correlation.
    '''
    r1 = r1.shift(-r1.start)
    r2 = r2.shift(-r2.start)
    r1, r2 = cls.equalize(r1, r2)
    x_axis = ArrayAxis(start=-r1.end, end=r1.end, sample=r1.sample)
    return x_axis, np.correlate(r1.y, r2.y, "full")


def convolve(cls: Type["Relation"], r1: "Relation",
             r2: "Relation") -> Tuple[XAxis, np.ndarray]:
    '''Convolution.

    The method by which the convolution is performed.
    Using the `numpy.convolve` function.

    Args:
        cls (Type[&quot;Relation&quot;]): class to use equalization of two arrays.
        r1 (Relation): first function y.
        r2 (Relation): second function y.

    Returns:
        Tuple[XAxis, np.ndarray]: result of convolution.
    '''
    r1 = r1.shift(-r1.start)
    r2 = r2.shift(-r2.start)
    r1, r2 = cls.equalize(r1, r2)
    x_axis = ArrayAxis(start=-r1.end, end=r1.end, sample=r1.sample)
    return x_axis, np.convolve(r1.y, r2.y, "full")


# ==============================================================================

def _calculate_spectrum(
        time: TimeAxis, amplitude: np.ndarray) -> Tuple[FrequencyAxis, np.ndarray]:

    amplitude = np.append(
        amplitude[time.array >= 0.0], amplitude[time.array < 0.0])
    spectrum = np.fft.rfft(amplitude)
    np_frequency = np.fft.rfftfreq(
        amplitude.size, d=time.sample
    )

    frequency = get_array_axis_from_array(np_frequency, round_dx=False)
    return frequency, spectrum


def signal2spectrum(
    relation: 'Relation', is_start_zero=False
) -> Tuple[FrequencyAxis, np.ndarray]:
    '''Forward Fourier Transform.

    Method for converting a signal into a spectrum.
    Using the `numpy.fft.rfft` function.

    Args:
        relation (Relation): signal from which get spectrum.
        is_start_zero (bool, optional): Consider array started from zero time.
            Defaults to False.

    Returns:
        Tuple[FrequencyAxis, np.ndarray]: result transformation signal to
            spectrum.
    '''
    new_time = relation.x.copy()
    amplitude = relation.y.copy()
    if is_start_zero:
        return _calculate_spectrum(new_time, amplitude)

    if new_time.start > 0.0:
        new_time.start = 0.0
        amplitude = np.append(
            np.zeros(new_time.size - amplitude.size),
            amplitude)

    elif new_time.end < 0.0:

        new_time.end = 0.0
        amplitude = np.append(
            amplitude, np.zeros(
                new_time.size - amplitude.size))

    return _calculate_spectrum(new_time, amplitude)


def spectrum2signal(
    relation: 'Relation', time_start: float = None
) -> Tuple[TimeAxis, np.ndarray]:
    '''Inverse Fourier Transform.

    Method for converting a spectrum into a signal.
    Using `numpy.ifft` function.

    Args:
        relation (Relation): spectrum of signal.

        time_start (float, optional): default fft convert to 0. time. Maybe you
            want another start of time. Defaults to None.

    Returns:
        Tuple[TimeAxis, np.ndarray]: result transformation spectrogram to signal.
    '''

    spectrum = relation.y.copy()
    frequency = relation.x.copy()

    amplitude = np.fft.irfft(spectrum)  # type: np.ndarray
    dt = 1 / (2 * (frequency.end - frequency.start))

    if time_start is None:
        time = ArrayAxis(0., (amplitude.size - 1) * dt, dt)

    else:
        time = ArrayAxis(time_start, time_start +
                         (amplitude.size - 1) * dt, dt)

        amplitude = np.append(
            amplitude[time.array >= 0.0], amplitude[time.array < 0.0])

    return time, amplitude


def integrate_function(
    function: Callable[[np.ndarray], np.ndarray], x: ArrayAxis
) -> Tuple[ArrayAxis, np.ndarray]:
    '''Integration function y(x).

    The method by which the integration function is performed. Integration across
    the entire function. Get the expected integrated array function.
    Integration of function, using `scipy.integrate.quad` function.

    Args:
        function (Callable[[x], y]): function is describing
            changes frequency from time.

        x (np.ndarray): time array.

    Returns:
        Relation: result of integration function.
    '''

    result = np.append(
        [0.0],
        np.array(
            [2 * np.pi * quad_integrate_function(function, x.start, x_element)[0]
             for x_element in x.array[1:]]
        ),
    )
    return x, result
