from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from scipy.signal import hilbert, spectrogram  # type: ignore

from .. import relation, signal, spectrum
from ..axis import ArrayAxis
from ..config import base_config
from ..exc import BadInputError
from ..help_types import ArrayLike, Frequency, Time, Envelope, Spectrogram, Ftat

Ftatr = Union["relation.Relation", Ftat]
'''The Representation of object from which will be extracted frequency
modulation or amplitude modulation.'''

CallFtatMethod = Callable[[
    "spectrum.Spectrum"], Tuple[Time, Frequency, Envelope]]
'''Method is used to extract frequency and amplitude modulation from spectrum.'''


def simple_freq2time(
        spectrum: "spectrum.Spectrum") -> Tuple[Time, Frequency, Envelope]:
    '''The simple method to extract Frequency modulation from a prior spectrum.

    The amplitude modulation is constant.

    Args:
        spectrum (Spectrum): instance signal of 'Spectrum'

    Returns:
        Tuple[Time, Frequency, Envelope]: simple representation Frequency
            modulation from a prior spectrum.
    '''

    amplitude_spectrum = spectrum.get_amp_spectrum()
    frequency = amplitude_spectrum.array
    n_spec = amplitude_spectrum.y**2
    time = np.append(
        [0.0], ((n_spec[1:] + n_spec[:-1]) /
                (frequency[1:] - frequency[:-1])).cumsum()
    )
    coefficient = amplitude_spectrum.get_norm()
    amplitude_modulation = coefficient * np.ones(len(time))
    return time, frequency, amplitude_modulation


class InterpolateArray:
    '''Class to interpolate array.

    Use to stretch data to new axis.
    '''

    def __init__(
        self,
        x: Union[relation.Relation, ArrayAxis, ArrayLike],
        y: Optional[ArrayLike] = None
    ) -> None:
        '''Initialize InterpolateArray.

        Args:
            x (Union[relation.Relation, ArrayAxis, ArrayLike]): old array of x
                or old `Relation` instance.

            y (Optional[ArrayLike], optional): old array of y or None. Defaults to None.

        Raises:
            BadInputError: raise exception if will be not enough data.
        '''

        if isinstance(x, relation.Relation):
            self._x = x.x.array.copy()
            self._y = x.y.copy()
            return None

        elif y is None:
            raise BadInputError("Not enough data! y is absent")

        self._y = np.array(y)

        if isinstance(x, ArrayAxis):
            self._x = x.array.copy()
        else:
            self._x = np.array(x)

    def __call__(self, new_x: ArrayAxis) -> np.ndarray:
        '''Get new array of y.

        Stretch and interpolate old array y to get new array y.
        Args:
            new_x (ArrayAxis): new array of x.

        Returns:
            np.ndarray: new array of y.
        '''
        stretch_old_x = self._x * \
            ((new_x.end - new_x.start) /
             self._x[-1]) - self._x[0] + new_x.start

        new_y = base_config.Config. \
            interpolate_extrapolate_method(stretch_old_x, self._y)

        return new_y(new_x)


def convert_freq2time(
    spectrum: "spectrum.Spectrum", convert_method: CallFtatMethod
) -> Tuple[InterpolateArray, InterpolateArray]:
    '''Convert spectrum to function to create sweep.

        The method consists in obtaining the functions of Frequency and
        amplitude modulation.

    Args:
        spectrum (Spectrum): spectrum a prior data.

        convert_method (CallFtatMethod): method to convert spectrum into
            frequency-time and amplitude-time arrays.

    Returns:
        TupleInterpolateArray,InterpolateArray]:
            tuple of two function. First function describe changes frequency
            from time. Second function describe changes amplitude from time.
            This function will be use to create sweep signal.

    '''
    nT, f, a_t = convert_method(spectrum)
    return InterpolateArray(nT, f), InterpolateArray(nT, a_t)


def get_info_from_a_prior_data(
    time: Union["relation.Relation", ArrayAxis, ArrayLike],
    a_prior_data: Any,
    f_a_t_method: CallFtatMethod
) -> Tuple[InterpolateArray, InterpolateArray, "signal.Signal"]:
    '''Get the Frequency and amplitude modulation function from a prior data.

    Args:
        time (Any): time.

        a_prior_data (Any): a prior data.

        f_a_t_method (CallFtatMethod): method to convert a prior data for
            functions that will use to create sweep signal.

    Returns (TupleInterpolateArray,InterpolateArray, "Signal"]):
        return frequency-time and amplitude-time functions and a prior signal.
    '''

    if isinstance(time, spectrum.Spectrum):
        a_prior_signal = time.get_signal()
        a_prior_spectrum = time
    elif isinstance(time, relation.Relation):
        a_prior_signal = signal.Signal(time)
        a_prior_spectrum = a_prior_signal.get_spectrum()
    elif isinstance(a_prior_data, spectrum.Spectrum):
        a_prior_signal = a_prior_data.get_signal()
        a_prior_spectrum = a_prior_data
    elif isinstance(a_prior_data, signal.Signal):
        a_prior_signal = a_prior_data
        a_prior_spectrum = a_prior_data.get_spectrum()
    elif isinstance(a_prior_data, relation.Relation):
        a_prior_spectrum = spectrum.Spectrum(a_prior_data)
        a_prior_signal = a_prior_spectrum.get_signal()
    else:
        a_prior_signal = signal.Signal(time, a_prior_data)
        a_prior_spectrum = a_prior_signal.get_spectrum()

    f_t, a_t = convert_freq2time(a_prior_spectrum, f_a_t_method)

    return f_t, a_t, a_prior_signal


def get_info_from_ftat(
    t: Union[ArrayAxis, np.ndarray, None] = None,
    f_t: Union[Ftatr, InterpolateArray, None] = None,
    a_t: Union[Ftatr, InterpolateArray, None] = None
) -> Tuple[Optional[ArrayAxis],
           Union[Callable[[np.ndarray], np.ndarray], InterpolateArray],
           Union[Callable[[np.ndarray], np.ndarray], InterpolateArray]
           ]:
    '''Prepared information for 'UncalculatedSweep' instance.

    Args:
        t (Optional[np.ndarray]): time. Default None.
        f_t (Optional[Ftatr]): frequency-time function. Default None.
        a_t (Optional[Ftatr]): amplitude-time function. Default None.

    Returns:
        Tuple[Optional[np.ndarray],InterpolateArray,InterpolateArray]:
        return frequency-time and amplitude-time functions and a prior signal.
    '''
    if f_t is None:

        def linear_time(time: np.ndarray) -> np.ndarray:
            return time

        f_t = linear_time

    t1, f_t = _extract_x_t(t, f_t)

    if a_t is None:

        def const_one(time: np.ndarray) -> np.ndarray:
            return np.ones(len(time))

        a_t = const_one

    t2, a_t = _extract_x_t(t, a_t)

    if isinstance(t, np.ndarray):
        t = base_config.Config.get_array_axis_from_array_method(t, True)
    if t is None:
        if t1 is None and t2 is not None:
            t = t2
        elif t2 is None and t1 is not None:
            t = t1
        elif t2 is not None and t1 is not None:
            t = base_config.Config.get_common_x(t1, t2)

    return t, f_t, a_t


def _extract_x_t(
    t: Union[ArrayAxis, np.ndarray, None] = None,
    x_t: Union[Ftatr, InterpolateArray] = None,
) -> Tuple[
    Optional[ArrayAxis],
    Union[Callable[[np.ndarray], np.ndarray], InterpolateArray]
]:
    if isinstance(t, np.ndarray):
        t = base_config.Config.get_array_axis_from_array_method(t, False)

    b_x_t: Union[Callable[[np.ndarray], np.ndarray], InterpolateArray]
    if not callable(x_t):
        if isinstance(x_t, InterpolateArray):
            b_x_t = InterpolateArray
        elif isinstance(x_t, relation.Relation):
            b_x_t = InterpolateArray(x_t)
        else:
            if isinstance(t, np.ndarray):
                t = base_config.Config.get_array_axis_from_array_method(t)

            if isinstance(t, ArrayAxis) and isinstance(x_t, np.ndarray):
                if t.size != x_t.size:
                    calc_t = ArrayAxis(
                        t.start, t.end, (t.start - t.end) / (x_t.size - 1))

                    interpolate_x_t = base_config.Config. \
                        interpolate_extrapolate_method(calc_t.array, x_t)
                    x_t = interpolate_x_t(t)
                b_x_t = InterpolateArray(t, x_t)
            else:
                raise BadInputError("Not enough data: t or x_t")
    else:
        b_x_t = x_t

    return t, b_x_t

# ======================================================================
# Info about sweep.


def get_spectrogram(sweep: "relation.Relation") -> Spectrogram:
    '''Function to get spectrogram of the sweep signal.

    Using the `scipy.signal.spectrogram` function.

    Args:
        sweep (relation.Relation): instance of sweep signal.

    Returns:
        Spectrogram: tuple of np.ndarray. The first element is time.
            The second is frequency. The third is matrix M x N of spectrogram.
    '''

    if sweep.y.size < 256:
        nperseg = sweep.y.size
    else:
        nperseg = 256
    frequency, spectrogram_time, spectrogram_ = spectrogram(
        sweep.y, 1 / (sweep.sample), nperseg=nperseg)

    return spectrogram_time, frequency, spectrogram_[::-1, ::]


def get_f_t(sweep: "relation.Relation") -> "relation.Relation":
    '''Get Time-Frequency function from sweep signal using the Hilbert
    transformation.

    Using the `scipy.signal.hilbert` function.

    Args:
        sweep (Relation): instance of sweep signal.

    Returns:
        Relation: instance `Relation`
    '''

    analytical_signal = hilbert(sweep.y)
    result = np.append(
        [0.0],
        np.diff(np.unwrap(np.angle(analytical_signal)))
        / (2.0 * np.pi)
        / sweep.sample,
    )
    return relation.Relation(sweep.x, result)


def get_a_t(sweep: "relation.Relation") -> "relation.Relation":
    '''Get envelop from sweep signal using the Hilbert transformation.

    Using the `scipy.signal.hilbert` function.

    Args:
        sweep (Relation): instance of sweep signal.

    Returns:
        Relation: instance `Relation`
    '''

    analytical_signal = hilbert(sweep.y)
    return relation.Relation(sweep.x, np.abs(analytical_signal))
