import numpy as np
from numpy.testing import assert_array_equal

from sweep_design.spectrum import Spectrum
from sweep_design.axis import ArrayAxis
from sweep_design.exc import ConvertingError, TypeFuncError
from sweep_design.relation import Relation
from sweep_design.signal import Signal

from .test_relation import WrapperTestRelation


class WrapperTestSignal:

    class BaseTestSignal(WrapperTestRelation.TestBaseRelation):

        def setUp(self) -> None:
            self.relation_class = Signal
            self.x_axis = ArrayAxis(start=0, end=0.5, sample=0.1)
            self.relation = self.relation_class(self.x_axis,
                                                [10.1, -5.231, 123., 0., 12.465, 5.])

            self.simple_relation = self.relation_class(self.x_axis,
                                                       [10, 20, 30, 40, 50, 60])

            self.x_axis_2 = ArrayAxis(start=0, end=0.6, sample=0.1)
            self.simple_second_relation = self.relation_class(self.x_axis_2,
                                                              [10, 20, 30, 40, 50, 60, 70])

        def test_math_operations(self):
            x = ArrayAxis(start=0, end=4, sample=1)
            y1 = np.array([10, 20, 30, 40, 50], dtype="float")

            y2 = np.array([2, 4, 6, 8, 10], dtype="float")

            r1 = self.relation_class(x, y1)
            r2 = self.relation_class(x, y2)
            operation = [
                "__add__",
                "__sub__",
                "__mul__",
                "__truediv__",
                "__pow__"]

            for m in operation:
                for k in [(r2, y2), (2, 2)]:
                    with self.subTest(f"Operation {m}, operation elements {k}", k=k, m=m):
                        self._math_check(r1, k[0], x, y1, k[1], m)

            with self.assertRaises(TypeFuncError):
                r1 + 'wrong type'

        def test_get_spectrum(self):
            spectrum = self.relation.get_spectrum()
            self.assertIsInstance(spectrum, Spectrum)
            self.assertIsNot(spectrum.x, self.x_axis)

        def test_get_reverse(self):

            reversed_signal = self.relation.get_reverse_signal()
            self.assertIsInstance(reversed_signal, self.relation_class)
            self.assertIsNot(reversed_signal.x, self.x_axis)
            assert_array_equal(reversed_signal.array, self.x_axis.array)

        def test_get_amplitude_spectrum(self):

            amplitude_spectrum = self.relation.get_amplitude_spectrum()
            self.assertIsInstance(amplitude_spectrum, Relation)
            self.assertIsNot(amplitude_spectrum.x, self.x_axis)

        def test_get_phase_spectrum(self):

            amplitude_spectrum = self.relation.get_phase_spectrum()
            self.assertIsInstance(amplitude_spectrum, Relation)
            self.assertIsNot(amplitude_spectrum.x, self.x_axis)

        def test_get_signal_from_spectrum_from_signal(self):
            signal = self.simple_relation.get_spectrum().get_signal()
            self.assertIsInstance(signal, Signal)

            signal = self.simple_second_relation.get_spectrum().get_signal()
            self.assertIsInstance(signal, Signal)

        def test_shift_signal(self):
            dt = 0.1
            time = ArrayAxis(0, 1, 0.1)
            data = np.zeros(time.size)
            data[0] = 1
            delta = Signal(time, data)

            shift1 = delta.shift(dt)
            self.assertEqual(shift1[dt][1], 1)

            shift2 = delta.shift(10 * dt)
            self.assertEqual(shift2[10 * dt][1], 1)


class TestSignal(WrapperTestSignal.BaseTestSignal):
    pass
