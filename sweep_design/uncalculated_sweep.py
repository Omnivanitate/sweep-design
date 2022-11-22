import logging
from math import sqrt
from typing import Any, Callable, Optional, Union

import numpy as np

from .axis import ArrayAxis, get_array_axis_from_array
from .config.base_config import Config
from .config.sweep_config import SweepConfig
from .defaults.sweep_methods import (CallFtatMethod, Ftatr, InterpolateArray,
                                     get_info_from_a_prior_data,
                                     get_info_from_ftat)
from .exc import BadInputError
from .relation import Relation
from .sweep import Sweep
from .types import ArrayLike


class UncalculatedSweep:
    '''The `UncalculatedSweep` class prepares for the calculation of the
    signal sweep (`Sweep`).

    The get_info_from_ftat function is used to extract the frequency versus
    `time` and amplitude versus `time` functions from the passed `frequency_time`
    and `amplitude_time` parameters.

    Analytic functions (`frequency_time` and `amplitude_time`) can be as array
    of numbers or as callable object(lambda function, common python functions and ect.)
    Example: frequency_time = lambda t: t*10+1, amplitude_time = lambda t: np.ones(t.size)

    Raises:
        BadInputError: raise exception when calling instance without parameter
        time or when time attribute is not created when instance initialized.
    '''

    def __init__(
        self,
        time: Union[ArrayAxis, ArrayLike] = None,
        frequency_time: Union[Ftatr, ArrayLike, InterpolateArray] = None,
        amplitude_time: Union[Ftatr, ArrayLike, InterpolateArray] = None,
    ) -> None:
        '''Initialize instance of UncalculatedSweep.

        Args:
            time (ArrayLike, optional): The `ArrayAxis`, or  an ArrayLike object
            containing numbers(real or complex). Defaults to None.

            frequency_time (Union[Ftatr, ArrayLike], optional):
            This parameter, which describes changes in frequency over time,
            can be either an array_like, or an object from which an instance
            of the `Relation` class will be created, or an instance of the
            `Relation` class, or a callable object that returns a numeric sequence.
            If `None`, then the linear function f = t will be used. Defaults to None.

            amplitude_time (Union[Ftatr, ArrayLike], optional):
            This parameter, which describes changes in amplitude modulation
            over time, can be either an array_like, or an object from which
            an instance of the `Relation` class will be created, or an instance
            of the `Relation` class, or a callable object that
            returns a numeric sequence.
            If `None`, then the function will be assumed to be constant
            and equal to 1. Defaults to None.
        '''

        self._integrate_function_default = Config.integrate_function_method

        # time = time if time is None else time
        if not (isinstance(time, ArrayAxis) or time is None):
            time = get_array_axis_from_array(time)

        if not (
            isinstance(
                frequency_time,
                (np.ndarray,
                 Relation,
                 InterpolateArray))
            or callable(frequency_time)
            or frequency_time is None
        ):
            frequency_time = np.array(frequency_time)
        if not (
            isinstance(
                amplitude_time,
                (np.ndarray,
                 Relation,
                 InterpolateArray))
            or callable(amplitude_time)
            or amplitude_time is None
        ):
            amplitude_time = np.array(amplitude_time)

        (
            self._time,
            self._frequency_time,
            self._amplitude_time,
        ) = get_info_from_ftat(time, frequency_time, amplitude_time)

    def __call__(self,
                 time: Union[ArrayAxis, ArrayLike] = None,
                 tht0=0.0
                 ) -> Sweep:
        '''Calling an instance with appropriate to calculate the sweep signal.

        If time is not passed or equals None, then the time sequence created
        when the class instance was initialized will be used.

        Args:
            time (Union[ArrayAxis, ArrayLike], optional): The number sequence
            determines the time. Defaults to None.

            tht0 (float, optional): Zero phase. Defaults to 0.0.

        Raises:
            BadInputError: raise exception when calling instance without parameter
            time when time attribute is not created when instance initialized.

        Returns:
            Sweep: an instance of the Sweep class - the calculated sweep signal.
        '''

        logging.info(
            "Calling uncalculated sweep.\n"
            "with params:\nfrequency_time={0}\namplitude_time={1}\ntime={2}"
            "".format(self._frequency_time, self._amplitude_time, time)
        )

        if time is None and self._time is None:
            raise BadInputError("Not enough data: time")

        elif time is not None:
            if isinstance(time, ArrayAxis):
                calc_time = time
            else:
                calc_time = get_array_axis_from_array(time)
        elif time is None and self._time is not None:
            calc_time = self._time

        if isinstance(self._frequency_time, InterpolateArray):
            tht = self._array_tht(self._frequency_time(calc_time)[1])
            frequency_time = Relation(*self._frequency_time(calc_time))
        else:
            tht = self._func_tht(self._frequency_time)
            frequency_time = Relation(
                calc_time, self._frequency_time(
                    calc_time.array))

        if isinstance(self._amplitude_time, InterpolateArray):
            amplitude = self._amplitude_time(calc_time)[1]
        else:
            amplitude = self._amplitude_time(calc_time.array)

        sweep = amplitude * \
            np.sin(tht(calc_time).y + tht0)

        amplitude_time = Relation(
            calc_time, amplitude
        )

        return Sweep(
            time=calc_time,
            amplitude=sweep,
            frequency_time=frequency_time,
            amplitude_time=amplitude_time,
        )

    def _func_tht(
        self, frequency_time: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[ArrayAxis], Relation]:
        """Functional representation of angular sweep."""

        def result(time: ArrayAxis) -> Relation:
            return Relation(
                *self._integrate_function_default(frequency_time, time))
        return result

    def _array_tht(
        self, frequency_time: np.ndarray
    ) -> Callable[[ArrayAxis], Relation]:
        """Angular sweep represented by a numerical sequence."""

        def wrapper(time: ArrayAxis) -> Relation:
            frequency_time_relation = Relation(time, frequency_time)
            result = np.append([0.0], 2 * np.pi *
                               frequency_time_relation.integrate().y)
            return Relation(time, result)

        return wrapper


class ApriorUncalculatedSweep(UncalculatedSweep):
    '''`ApriorUncalculatedSweep`

    Class for constructing a sweep signal from a priori data (from another
    signal or spectrum).

    The calculation of the change in frequency with time (`frequency_time`)
    and the amplitude envelope with time (`amplitude_time`) will depend on
    the a priori data (`aprior_data`) and on the method (`ftat_method`)
    by which they will be calculated.

    The extracted frequency over time (`frequency_time`) and the amplitude
    envelope over time (`amplitude_time`) will be send to the
    `UncalculatedSweep` constructor.
    '''

    def __init__(
        self,
        time: Any = None,
        a_prior_data: Any = None,
        ftat_method: Optional[CallFtatMethod] = None,
    ) -> None:
        '''Configuring an instance to create a sweep from a aprior data.

        Args:
            time (Any, optional): _description_. Defaults to None.

            a_prior_data (Any, optional): _description_. Defaults to None.

            ftat_method (CallFtatMethod, optional): If the conversion method
            (`ftat_method`) is not defined or `None`, then the method is
            taken from the `SweepConfig` class `freq2time`  Method can be
            overridden if necessary
            (`sweep_design.math_signals.config.SweepConfig.freq2time`).
            Defaults to None.
        '''
        if ftat_method is None:
            ftat_method = SweepConfig.freq2time

        (
            frequency_time,
            amplitude_time,
            self._a_prior_signal,
        ) = get_info_from_a_prior_data(time, a_prior_data, ftat_method)
        super().__init__(time, frequency_time, amplitude_time)

    def __call__(
        self,
        time: Union[ArrayAxis, ArrayLike] = None,
        tht0=0.0,
        is_normalize=True
    ) -> Sweep:
        '''Calculate the sweep and normalize it.

        Args:
            time (Union[ArrayAxis, ArrayLike], optional): The number sequence
            determines the time. Defaults to None.

            tht0 (float, optional): Zero phase. Defaults to 0.0.

            is_normalize (bool, optional): Use normalization a prior data for
            sweep signal. Defaults to True.

        Returns:
            Sweep: an instance of the Sweep class - the calculated sweep signal
            from a prior data.
        '''

        sweep = super().__call__(time=time, tht0=tht0)

        if is_normalize:
            norm_sweep = sweep.get_norm()
            norm_a_prior = self._a_prior_signal.get_norm()
            norm = sqrt(norm_a_prior) / sqrt(norm_sweep)
            sweep.amplitude_time *= norm
            sweep *= norm

        sweep.a_prior_signal = self._a_prior_signal
        return sweep
