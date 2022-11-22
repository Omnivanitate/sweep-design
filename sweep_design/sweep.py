from typing import Union

import numpy as np

from .axis import ArrayAxis, get_array_axis_from_array
from .config.sweep_config import SweepConfig
from .core import RelationProtocol
from .defaults.sweep_methods import Spectrogram as DataSpectrogram
from .relation import Relation
from .signal import Signal
from .spectrogram import Spectrogram
from .types import ArrayLike


class Sweep(Signal):
    '''Class `Sweep`.

    A class for analyzing changes in a signal over time, inherited from
    the `Signal` class.

    For analysis, you can use not only the sweep signal, but also other
    signals for which the spectrogram needs to be considered.

    When creating an instance of the class, the spectrogram is calculated.
    The method used to calculate the spectrogram is defined in the
    `SweepConfig` class. You can override it with your own.

    If the frequency vs. time and amplitude vs. time functions have not
    been passed, they are also calculated, the `get_frequency_time`,
    `get_amplitude_time` methods defined in the `SweepConfig` class are used.

    Perform the same operations as for the inherited class.
    '''

    def __init__(
        self,
        time: Union[RelationProtocol, ArrayAxis, ArrayLike],
        amplitude: ArrayLike = None,
        frequency_time: Relation = None,
        amplitude_time: Relation = None,
        a_prior_signal: Signal = None,
    ) -> None:
        '''Initialize sweep instance.

        Args:
            time (Union[RelationProtocol, ArrayAxis, ArrayLike]):
            The Relation class, or a class derived from the Relation class,
            or ArrayAxis, or an array_like object containing
            numbers(real or complex).

            amplitude (ArrayLike, optional): `None` or array_like object
            containing real or complex numbers. Defaults to None.

            frequency_time (Relation, optional):
            This parameter describes the change in frequency versus time of the
            transmitted signal. Defaults to None.

            amplitude_time (Relation, optional): This parameter describes
            the change in amplitude envelop of signal from the time of the
            transmitted signal. Defaults to None.


            a_prior_signal (Signal, optional): The signal used to create
            the sweep signal. Defaults to None.

        '''

        super().__init__(time, amplitude)

        self.frequency_time = (
            frequency_time
            if frequency_time is not None
            else SweepConfig.get_f_t(self)
        )
        self.amplitude_time = (
            amplitude_time
            if amplitude_time is not None
            else SweepConfig.get_a_t(self)
        )

        spectrogram = SweepConfig.spectrogram_method(self)

        self.spectrogram = _get_spectrogram(spectrogram)

        self.a_prior_signal = a_prior_signal


def _get_spectrogram(spectrogram: DataSpectrogram) -> Spectrogram:
    spectrogram_ = spectrogram[2]
    if spectrogram[0].size < 2:
        time = np.append(spectrogram[0]
                         [0], spectrogram[0][0] + 1)
        spectrogram_ = np.concatenate(
            (
                spectrogram_,
                np.zeros((spectrogram_.shape[0], 1))
            ),
            axis=1)
    else:
        time = spectrogram[0]

    if spectrogram[1].size < 2:
        frequency = np.append(
            spectrogram[1][0], spectrogram[1][0] + 1)
        spectrogram_ = np.concatenate(
            (
                spectrogram_,
                np.zeros((1, spectrogram_.shape[1]))
            ),
            axis=0)
    else:
        frequency = spectrogram[1]

    return Spectrogram(
        time=get_array_axis_from_array(time),
        frequency=get_array_axis_from_array(frequency),
        spectrogram=spectrogram[2],
    )
