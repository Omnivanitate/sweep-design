"""This is where default methods are defined."""
from typing import TYPE_CHECKING, Callable, Tuple, Type, Union

import numpy as np
import scipy  # type: ignore
from packaging import version
from scipy.interpolate import interp1d  # type: ignore

from ..axis import ArrayAxis, get_array_axis_from_array
from ..core import MathOperation
from ..types import Number

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
Y = np.ndarray
FrequencyAxis = ArrayAxis
Spectrum = np.ndarray
TimeAxis = ArrayAxis
Amplitude = np.ndarray

if TYPE_CHECKING:
    from ..relation import Relation


def math_operation(
    x: ArrayAxis,
    y1: np.ndarray,
    y2: Union[np.ndarray, Number],
    name_operation: MathOperation,
) -> Tuple[XAxis, Y]:
    """Math operations.

    Using numpy math operations.
    """
    if name_operation == MathOperation.POW:
        y = np.abs(y1).__getattribute__(name_operation.value)(y2) * np.sign(y1)
    else:
        y = y1.__getattribute__(name_operation.value)(y2)

    return x, y


def one_integrate(relation: 'Relation') -> float:
    """Integration.

    Taking the integral on a segment. Return of the area under the graph.
    """
    x, y = relation.get_data()
    return integration(y, x)


def integrate(relation: 'Relation') -> Tuple[XAxis, Y]:
    """Integration.

    Using the scipy.integrate.cumtrapz function.
    """
    array_axis = relation.x.copy()
    dx = array_axis.sample
    array_axis.start = array_axis.start + array_axis.sample
    return array_axis, cumulative_integration(relation.y) * (dx)


def differentiate(relation: 'Relation') -> Tuple[XAxis, Y]:
    """Differentiation.

    Using the numpy.diff function."""
    array_axis = relation.x.copy()
    dx = array_axis.sample
    array_axis.start = array_axis.start + array_axis.sample / 2
    array_axis.end = array_axis.end - array_axis.sample / 2
    return array_axis, np.diff(relation.y) / (dx)


def interpolate_extrapolate(
    x: XAxis, y: Y, bounds_error=False, fill_value=0.0
) -> Callable[[ArrayAxis], Tuple[XAxis, Y]]:
    """Interpolation.

    Using the scipy.interpolate.interp1d function.
    Returning function of interpolation.
    """
    def wrapper(new_x: ArrayAxis) -> Tuple[XAxis, Y]:
        new_y = interp1d(x.array, y, bounds_error=bounds_error,
                         fill_value=fill_value)(new_x.array)
        return new_x, new_y

    return wrapper


def get_common_x(x1: XAxis, x2: XAxis) -> XAxis:
    """Specifies the overall x-axis.

    Finds the general sample rate and beginning and end of sequence.
    """
    dx1 = x1.sample
    dx2 = x2.sample

    dx = dx1 if dx1 <= dx2 else dx2
    x_start = x1.start if x1.start <= x2.start else x2.start
    x_end = x1.end if x1.end >= x2.end else x2.end
    return ArrayAxis(start=x_start, end=x_end, sample=dx)


def correlate(cls: Type["Relation"], r1: "Relation",
              r2: "Relation") -> Tuple[XAxis, np.ndarray]:
    """Correlation.

    Using the numpy.correlate function.
    """
    r1 = r1.shift(-r1.start)
    r2 = r2.shift(-r2.start)
    r1, r2 = cls.equalize(r1, r2)
    x_axis = ArrayAxis(start=-r1.end, end=r1.end, sample=r1.sample)
    return x_axis, np.correlate(r1.y, r2.y, "full")


def convolve(cls: Type["Relation"], r1: "Relation",
             r2: "Relation") -> Tuple[XAxis, np.ndarray]:
    """Convolution.

    Using the numpy.convlove function.
    """
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
    """Forward Fourier Transform.

    Using the numpy.fft.rfft function.
    """
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
    """Inverse Fourier Transform.

    Using the numpy.fft.irfft function.
    """

    spectrum = relation.y.copy()
    frequency = relation.x.copy()
    amplitude = np.fft.irfft(spectrum)  # type: np.ndarray

    if time_start is None:
        np_time = np.linspace(
            0,
            (amplitude.size - 1) / (2 * (frequency.end - frequency.start)),
            amplitude.size,
        )
    else:
        np_time = np.linspace(
            time_start,
            time_start + (amplitude.size - 1) /
            (2 * (frequency.end - frequency.start)),
            amplitude.size,
        )
        amplitude = np.append(
            amplitude[np_time >= 0.0], amplitude[np_time < 0.0])

    time = get_array_axis_from_array(np_time)

    return time, amplitude


def integrate_function(
    function: Callable[[np.ndarray], np.ndarray], x: ArrayAxis
) -> Tuple[ArrayAxis, np.ndarray]:
    '''Integration function y(x).

    Integration of function, using scipy.integrate.quad function.

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
