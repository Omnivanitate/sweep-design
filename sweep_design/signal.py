from typing import Optional, Type, TypeVar, Union

import numpy as np  # type: ignore

from . import spectrum
from .axis import ArrayAxis
from .config.base_config import Config
from .core import RelationProtocol
from .exc import ConvertingError
from .relation import Relation
from .help_types import ArrayLike, Number

S = TypeVar("S", bound="Signal")
'''Instance of `Signal`.'''

SSPR = Union["spectrum.Spectrum", "Signal", Relation]
'''Instance of `Signal` or `Spectrum` or `Relation`.'''

SSPRN = Union["spectrum.Spectrum", "Signal", Relation, Number]
'''Instance of `Signal` or `Spectrum` or `Relation` or `Number`.'''


class Signal(Relation):
    '''Class describing some kind of signal.

    The `Signal` class inherits the `Relation` class.

    Each signal can be converted into a `spectrum.Spectrum` using method `get_spectrum`.
    To convert the signal into a spectrum, the method defined in the `Config`
    class is used. (Config.signal2spectrum_method). Current method can be
    overridden by own in `Config` class.

    When performing arithmetic operations on instances of the spectrum.Spectrum class,
    an instance of the `Signal` class will be extracted from
    the `spectrum.Spectrum` instance, and arithmetic operations will be performed
    on this instance. An instance of `Relation` class will be converted into
    the instance of `Signal` class.
    '''

    def __init__(
        self,
        time: Union[RelationProtocol, ArrayAxis, ArrayLike],
        amplitude: ArrayLike = None,
        **kwargs
    ) -> None:
        '''Initialization of instance of `Signal`.

        Args:
            time (Union[RelationProtocol, ArrayAxis, ArrayLike]): An instance
                of Relation class or inherited from it, or ArrayLike, or array_like
                object containing numbers (real or complex).

            amplitude (ArrayLike, optional): None or array_like object
                containing numbers (real or complex). Defaults to None.
        '''

        self._signal2spectrum_method_default = Config.signal2spectrum_method
        super().__init__(time, amplitude, **kwargs)
        self._spectrum: Optional[spectrum.Spectrum] = None

    @property
    def time(self) -> ArrayAxis:
        '''Time array axis.

        Equal to property `x`.

        Returns:
            ArrayAxis: time array axis.
        '''
        return self.x

    @property
    def amplitude(self) -> np.ndarray:
        '''Amplitude array.

        Equal to property 'y'

        Returns:
            np.ndarray: amplitude array.
        '''
        return self.y

    def get_spectrum(self, recalculate=False,
                     is_start_zero=False) -> "spectrum.Spectrum":
        '''Get spectrum from signal.

        Args:
            recalculate (bool, optional): If `True` then the spectrum will be
                calculated again, else false, then the saved one will be used.
                Defaults to `False`.

            is_start_zero (bool, optional): If True then the signal will be
                shifted to zero. Defaults to `False`.

        Returns:
            spectrum.Spectrum: instance of `spectrum.Spectrum` described this `Signal`.
        '''

        if self._spectrum is None or recalculate:
            f, a = self._signal2spectrum_method_default(
                self, is_start_zero)
            self._spectrum = spectrum.Spectrum(f, a)

        return self._spectrum

    def get_amplitude_spectrum(
        self, recalculate=False, is_start_zero=False
    ) -> Relation:
        '''Extract amplitude spectrum from `spectrum.Spectrum`.

        Method `get_spectrum` is used to get instance of `spectrum.Spectrum`.
        The amplitude spectrum is calculated from it using `get_amp_spectrum`
        method.

        Args:
            recalculate (bool, optional): If `True` then the spectrum will be
                calculated again, else false, then the saved one will be used.
                Defaults to `False`.

            is_start_zero (bool, optional): If True then the signal will be
                shifted to zero. Defaults to `False`.

        Returns:
            Relation: amplitude spectrum expected Relation instance.
        '''
        return self.get_spectrum(recalculate, is_start_zero).get_amp_spectrum()

    def get_phase_spectrum(self, recalculate=False,
                           is_start_zero=False) -> Relation:
        '''Extract amplitude spectrum from `spectrum.Spectrum`.

        Method `get_spectrum` is used to get instance of `spectrum.Spectrum`.
        The amplitude spectrum is calculated from it using `get_amp_spectrum`
        method.

        Args:
            recalculate (bool, optional): If `True` then the spectrum will be
                calculated again, else false, then the saved one will be used.
                Defaults to `False`.

            is_start_zero (bool, optional): If True then the signal will be
                shifted to zero. Defaults to `False`.

        Returns:
            Relation: amplitude spectrum expected Relation instance.
        '''
        return self.get_spectrum(
            recalculate, is_start_zero).get_phase_spectrum()

    def get_reverse_signal(
        self: S,
        percent: Union[float, int] = 5.0,
        subtract_phase: bool = True,
        frequency_start: float = None,
        frequency_end: float = None,
    ) -> S:
        '''Calculate reversed signal.

        Args:
            self (S): instance of Signal.

            percent (Union[float, int], optional): level of added white
                noise in percent. Defaults to 5.0.

            subtract_phase (bool, optional): If True performs phase subtraction,
                If False succeeds, add the phase. Defaults to True.

            frequency_start (float, optional): The start frequency.
                Defaults to None.

            frequency_end (float, optional): The end frequency.
                Defaults to None.

        Returns:
            S: instance of Signal.
        '''

        signal = (
            self.get_spectrum()
            .get_reverse_filter(
                percent,
                subtract_phase,
                frequency_start,
                frequency_end
            )
            .get_signal()
        )

        return type(self)(signal)

    def add_phase(self: S, other: SSPR) -> S:
        '''Add phase to signal.

        Args:
            self (S): instance of `Signal`.

            other (SSPR): Extracting the `spectrum.Spectrum` from the object and adding
                the phase `spectrum.Spectrum` to the `Signal`.

        Returns:
            S: new instance of `Signal`.
        '''
        sp_other = spectrum._input2spectrum(other)
        self_spectrum = self.get_spectrum()
        new_spectrum = spectrum.Spectrum.get_spectrum_from_amp_phase(
            self_spectrum.get_amp_spectrum(),
            self_spectrum.get_phase_spectrum() + sp_other.get_phase_spectrum(),
        )
        return type(self)(new_spectrum.get_signal())

    def sub_phase(self: S, other: SSPR) -> S:
        '''Subtrack phase from signal.

        Args:
            self (S): instance of Signal

            other (SSPR): Extracting the spectrum from the object and subtrack
                the phase spectrum from the signal.

        Returns:
            S: new instance of Signal.
        '''
        sp_other = spectrum._input2spectrum(other)
        self_spectrum = self.get_spectrum()
        new_spectrum = spectrum.Spectrum.get_spectrum_from_amp_phase(
            self_spectrum.get_amp_spectrum(),
            self_spectrum.get_phase_spectrum() - sp_other.get_phase_spectrum(),
        )
        return type(self)(new_spectrum.get_signal())

    @classmethod
    def convolve(cls: Type[S], r1: SSPR, r2: SSPR) -> S:
        '''Convolution of two instances of `Relation` and return new instance of
        `Signal`. Instances of `spectrum.Spectrum` will be converted to `Signal`

        Args:
            cls (Type[S]): `Signal` class.
            r1 (SSPR): instance of `Relation` or subclass of 'Relation'
            r2 (SSPR): instance of `Relation` or subclass of 'Relation'

        Returns:
            S: new instance of Signal.
        '''
        s_r1 = _inp2signal(r1)
        s_r2 = _inp2signal(r2)
        return cls(super().convolve(s_r1, s_r2))

    @classmethod
    def correlate(cls: Type[S], r1: SSPR, r2: SSPR) -> S:
        '''Correlation of two instances of `Relation` and return new instance of
        `Signal`. Instance of `spectrum.Spectrum` will be converted to `Signal`

        Args:
            cls (Type[S]): `Signal` class.
            r1 (SSPR): instance of `Relation` or subclass of 'Relation'
            r2 (SSPR): instance of `Relation` or subclass of 'Relation'

        Returns:
            S: new instance of Signal.
        '''
        s_r1 = _inp2signal(r1)
        s_r2 = _inp2signal(r2)
        return cls(super().correlate(s_r1, s_r2))

    def __add__(self: S, a: SSPRN) -> S:
        s_a = _inp2signal_operation(a)
        return super().__add__(s_a)

    def __sub__(self: S, a: SSPRN) -> S:
        s_a = _inp2signal_operation(a)
        return super().__sub__(s_a)

    def __mul__(self: S, a: SSPRN) -> S:
        s_a = _inp2signal_operation(a)
        return super().__mul__(s_a)

    def __truediv__(self: S, a: SSPRN) -> S:
        s_a = _inp2signal_operation(a)
        return super().__truediv__(s_a)

    def __pow__(self: S, a: SSPRN) -> S:
        s_a = _inp2signal_operation(a)
        return super().__pow__(s_a)


def _inp2signal_operation(inp: SSPRN) -> Union[Relation, Signal, Number]:
    if isinstance(inp, spectrum.Spectrum):
        return inp.get_signal()
    else:
        return inp


def _inp2signal(inp: SSPR) -> Signal:
    if isinstance(inp, spectrum.Spectrum):
        return inp.get_signal()
    elif isinstance(inp, Signal):
        return inp
    elif isinstance(inp, Relation):
        return Signal(inp)
    else:
        raise ConvertingError(type(inp), Signal)
