from typing import Optional, Type, TypeVar, Union

import numpy as np

from . import signal
from .axis import ArrayAxis
from .config.base_config import Config
from .core import RelationProtocol
from .exc import ConvertingError
from .relation import Relation
from .help_types import ArrayLike, Number

SP = TypeVar("SP", bound="Spectrum")
'''Instance of `Spectrum`'''

SSPR = Union["Spectrum", "signal.Signal", Relation]
'''Instance of Spectrum or Signal or Relation'''

SSPRN = Union["Spectrum", "signal.Signal", Relation, Number]
'''Instance of Spectrum or Signal or Relation or Number'''


class Spectrum(Relation):
    '''A class that describes the spectrum of a signal.

    The `Spectrum` class derived from the `Relation` class.

    Each 'Spectrum' can be converted into a `Signal` using method `get_spectrum`
    To convert the `Spectrum` into a `Signal`, the method defined in the `Config`
    class is used. (`Config.spectrum2signal_method`). Current method can be
    overridden by own in `Config` class.

    When performing arithmetic operations on instances of the `Signal` class,
    an instance of the `Spectrum` class will be extracted from
    the `Signal` instance, and arithmetic operations will be performed
    on this instance. An instance of `Relation` class will be converted into
    the instance of `Spectrum` class.

    '''

    def __init__(
        self,
        frequency: Union[RelationProtocol, ArrayAxis, ArrayLike],
        spectrum_amplitude: ArrayLike = None,
        signal: Optional["signal.Signal"] = None
    ) -> None:
        '''Initialization of instance of `Spectrum`.

        Args:
            frequency (Union[RelationProtocol, ArrayAxis, ArrayLike]):
                An instance of `Relation` class or inherited from it,
                or `ArrayAxis` instance, or array_like object containing
                numbers (real or complex).

            spectrum_amplitude (ArrayLike, optional):
                None or array_like object containing numbers (real or complex).
                Defaults to None.

        '''
        super().__init__(frequency, spectrum_amplitude)
        self._spectrum2signal_method_default = Config.spectrum2signal_method
        self._signal = signal

    @property
    def frequency(self) -> ArrayAxis:
        '''Frequency array axis.

        Equal to property `x`.
        Returns:
            ArrayAxis: frequency array axis.
        '''
        return self.x

    @property
    def amplitude(self) -> np.ndarray:
        '''Spectrum amplitude array.

        Equal to property `y`.
        Returns:
            np.ndarray: spectrum amplitude array.
        '''
        return self.y

    def get_signal(
        self,
        time: Optional[Union[ArrayAxis, int]] = None,
        start_time: float = None
    ) -> "signal.Signal":
        '''Get signal from spectrum.

        Compute the signal from the spectrum.

        Args:
            time (ArrayAxis, int, optional): Define time to calculate
            signal. Defaults to None.

            start_time (float, optional): If True then the signal will be
                shifted to zero. Defaults to `False`.

        Returns:
            signal.Signal: instance of `Signal` described this `Spectrum`.
        '''

        if self._signal is None or time:

            time, amplitude = self._spectrum2signal_method_default(
                self, time, start_time
            )
            self._signal = signal.Signal(time, amplitude, self)

        return self._signal

    def get_amp_spectrum(self: SP) -> Relation:
        '''Get amplitude spectrum.

        Calculate the relationship between the frequency and the absolute
        value of the spectrum amplitude.

        Args:
            self (SP): instance of Spectrum.

        Returns:
            Relation: new instance of Relation.
        '''
        return Relation(self.x.copy(), np.abs(self.y))

    def get_phase_spectrum(self: SP) -> Relation:
        '''Calculate the relationship between frequency and phase of the spectrum.

        Args:
            self (SP): instance of Spectrum.

        Returns:
            Relation: new instance of Relation.
        '''

        return Relation(self.x.copy(), np.unwrap(np.angle(self.y)))

    def get_reverse_filter(
        self: SP,
        percent: Union[float, int] = 5.0,
        subtrack_phase=True,
        frequency_start: float = None,
        frequency_end: float = None,
    ) -> SP:
        '''Calculate filter of reversed signal.

        Args:
            self (SP): instance of Spectrum.

            percent (Union[float, int], optional): level of added white noise
                in percent. Defaults to 5.0.

            subtrack_phase (bool, optional): If True performs phase subtraction.
                If False succeeds, add the phase. Defaults to True.

            frequency_start (float, optional): The start frequency. Defaults to None.
            frequency_end (float, optional): The end frequency. Defaults to None.

        Returns:
            SP: new instance of Spectrum.
        '''

        spectrum = self.select_data(frequency_start, frequency_end)
        abs_spectrum = spectrum.get_amp_spectrum()
        abs_spectrum = abs_spectrum + abs_spectrum.max() * percent / 100
        reversed_abs_spectrum = 1 / abs_spectrum

        if subtrack_phase:
            phase_spectrum = -1 * spectrum.get_phase_spectrum()
        else:
            phase_spectrum = 1 * spectrum.get_phase_spectrum()

        result_spectrum = type(self).get_spectrum_from_amp_phase(
            reversed_abs_spectrum, phase_spectrum
        )
        return result_spectrum

    def add_phase(self: SP, other: SSPR) -> SP:
        '''Add phase to spectrum.

        Args:
            self (SP): instance of `Spectrum`

            other (SSPR): Extracting the `Spectrum` from the object and adding
                the phase `Spectrum` to the `Spectrum`.

        Returns:
            `SP`: new instance of `Spectrum`.
        '''

        sp_other = _input2spectrum(other)
        return type(self).get_spectrum_from_amp_phase(
            self.get_amp_spectrum(),
            self.get_phase_spectrum() + sp_other.get_phase_spectrum(),

        )

    def sub_phase(self: SP, other: SSPR) -> SP:
        '''Subtrack phase from spectrum.

        Args:
            self (SP): instance of `Spectrum`

            other (SSPR): Extracting the `Spectrum` from the object and subtrack
                the phase `Spectrum` from the `Spectrum`.

        Returns:
            `SP`: new instance of `Spectrum`.
        '''
        sp_other = _input2spectrum(other)
        return type(self).get_spectrum_from_amp_phase(
            self.get_amp_spectrum(),
            self.get_phase_spectrum() - sp_other.get_phase_spectrum(),

        )

    @classmethod
    def get_spectrum_from_amp_phase(
        cls: Type[SP], amplitude_spectrum: Relation, phase_spectrum: Relation
    ) -> SP:
        '''Calculate of the spectrum from the amplitude and phase spectrum.

        The spectrum is calculated through the amplitude and phase spectrum
        using the formula abs*exp(1j*phase).

        Args:
            cls (Type[SP]): Spectrum class.
            amplitude_spectrum (Relation): Amplitude spectrum is instance of 'Relation'.
            phase_spectrum (Relation): Phase spectrum is instance of 'Relation'.

        Returns:
            SP: new instance of `Spectrum`
        '''

        return cls(amplitude_spectrum *
                   ((1.0j * phase_spectrum).exp()))

    @classmethod
    def convolve(cls: Type[SP], r1: SSPR, r2: SSPR) -> SP:
        '''Convolution of two instances of `Relation` and return new instance of
        `Spectrum`. Instances of `Signal` will be converted to `Spectrum`

        Args:
            cls (Type[S]): `Signal` class.
            r1 (SSPR): instance of `Relation` or subclass of 'Relation'
            r2 (SSPR): instance of `Relation` or subclass of 'Relation'

        Returns:
            S: new instance of `Spectrum`.
        '''
        sp_r1 = _input2spectrum(r1)
        sp_r2 = _input2spectrum(r2)
        return super().convolve(sp_r1, sp_r2)

    @classmethod
    def correlate(cls: Type[SP], r1: SSPR, r2: SSPR) -> SP:
        '''Correlation of two instances of `Relation` and return new instance of
        `Spectrum`. Instances of `Signal` will be converted to `Spectrum`

        Args:
            cls (Type[S]): `Signal` class.
            r1 (SSPR): instance of `Relation` or subclass of 'Relation'
            r2 (SSPR): instance of `Relation` or subclass of 'Relation'

        Returns:
            S: new instance of `Spectrum`.
        '''
        sp_r1 = _input2spectrum(r1)
        sp_r2 = _input2spectrum(r2)
        return super().correlate(sp_r1, sp_r2)

    def __add__(self: SP, a: SSPRN) -> SP:
        r_a = _input2spectrum_operation(a)
        return super().__add__(r_a)

    def __sub__(self: SP, a: SSPRN) -> SP:
        r_a = _input2spectrum_operation(a)
        return super().__sub__(r_a)

    def __mul__(self: SP, a: SSPRN) -> SP:
        r_a = _input2spectrum_operation(a)
        return super().__mul__(r_a)

    def __truediv__(self: SP, a: SSPRN) -> SP:
        r_a = _input2spectrum_operation(a)
        return super().__truediv__(r_a)

    def __pow__(self: SP, a: SSPRN) -> SP:
        r_a = _input2spectrum_operation(a)
        return super().__pow__(r_a)


def _input2spectrum_operation(
        inp: SSPRN) -> Union[Relation, Spectrum, Number]:
    if isinstance(inp, signal.Signal):
        return inp.get_spectrum()
    elif isinstance(inp, (Spectrum, Relation, int, float, complex)):
        return inp
    else:
        raise ConvertingError(type(inp), Spectrum)


def _input2spectrum(inp: SSPR) -> Spectrum:
    if isinstance(inp, signal.Signal):
        return inp.get_spectrum()

    elif isinstance(inp, Spectrum):
        return inp

    elif isinstance(inp, Relation):
        return Spectrum(inp)
    else:
        raise ConvertingError(type(inp), Spectrum)
