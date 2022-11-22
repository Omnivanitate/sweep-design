import numpy as np
from numpy.testing import assert_array_equal

from sweep_design.spectrum import Spectrum
from sweep_design.axis import ArrayAxis
from sweep_design.exc import ConvertingError
from sweep_design.relation import Relation
from sweep_design.signal import Signal

from .test_relation import WrapperTestRelation


class TestSpectrum(WrapperTestRelation.TestBaseRelation):

    def setUp(self) -> None:
        self.relation_class = Spectrum
        self.x_axis = ArrayAxis(start=0, end=0.5, sample=0.1)
        self.relation = self.relation_class(self.x_axis,
                                            [10.1, -5.231, 123., 0., 12.465, 5.])

        self.simple_relation = self.relation_class(self.x_axis,
                                                   [10, 20, 30, 40, 50, 60])

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

        with self.assertRaises(ConvertingError):
            r1 + 'wrong type'

    def test_get_amplitude_spectrum(self):

        spectrum = self.relation
        amp = spectrum.get_amp_spectrum()
        self.assertIsInstance(amp, Relation)
        self.assertIsNot(amp.x, self.x_axis)
        assert_array_equal(amp.array, self.x_axis.array)

    def test_get_phase_spectrum(self):

        spectrum = self.relation
        phase = spectrum.get_phase_spectrum()
        self.assertIsInstance(phase, Relation)
        self.assertIsNot(phase.x, self.x_axis)
        assert_array_equal(phase.array, self.x_axis.array)

    def test_get_signal(self):

        spectrum = self.relation
        signal = spectrum.get_signal()
        self.assertIsInstance(signal, Signal)
        self.assertIsNot(signal.x, self.x_axis)

    def test_get_reversed_spectrum(self):

        spectrum = self.relation
        revers_spectrum = spectrum.get_reverse_filter()
        self.assertIsInstance(revers_spectrum, self.relation_class)
        self.assertIsNot(revers_spectrum.x, self.x_axis)
        assert_array_equal(revers_spectrum.array, self.x_axis.array)

    def test_get_spectrum_from_phase_and_amplitude(self):

        amplitude_spectrum = Relation(
            [0, 1, 2, 3, 4, 5],
            [1, 2, 4, 6, 8, 10]
        )
        phase_spectrum = Relation(
            [0, 1, 2, 3, 4, 5],
            [1, 2, 4, 6, 8, 10]
        )
        spectrum = self.relation_class.get_spectrum_from_amp_phase(
            amplitude_spectrum, phase_spectrum)
        self.assertIsInstance(spectrum, self.relation_class)
        self.assertIsNot(spectrum.x, amplitude_spectrum.x)
        self.assertIsNot(spectrum.x, phase_spectrum.x)
        assert_array_equal(spectrum.array, amplitude_spectrum.array)

    def test_add_phase(self):
        spectrum = self.relation_class(
            [0, 1, 2, 3, 4, 5],
            [1, 2, 4, 6, 8, 10]
        )
        phase_spectrum = self.relation_class(
            [0, 1, 2, 3, 4, 5],
            [1, 2, 4, 6, 8, 10]
        )
        result_spectrum = self.relation_class.add_phase(
            spectrum, phase_spectrum)
        self.assertIsInstance(result_spectrum, self.relation_class)
        self.assertIsNot(result_spectrum.x, phase_spectrum.x)
        assert_array_equal(result_spectrum.array, phase_spectrum.array)

    def test_subtrack_phase(self):
        spectrum = self.relation_class(
            [0, 1, 2, 3, 4, 5],
            [1, 2, 4, 6, 8, 10]
        )
        phase_spectrum = self.relation_class(
            [0, 1, 2, 3, 4, 5],
            [1, 2, 4, 6, 8, 10]
        )
        result_spectrum = self.relation_class.sub_phase(
            spectrum, phase_spectrum)
        self.assertIsInstance(result_spectrum, self.relation_class)
        self.assertIsNot(result_spectrum.x, phase_spectrum.x)
        assert_array_equal(result_spectrum.array, phase_spectrum.array)
