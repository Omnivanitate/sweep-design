import unittest

import numpy as np

from sweep_design.axis import ArrayAxis
from sweep_design.spectrum import Spectrum
from sweep_design.signal import Signal
from sweep_design.utility_functions.ftat_functions import proportional_freq2time, dwell
from sweep_design.utility_functions.emd_analyze import get_IMFs_ceemdan, get_IMFs_emd
from sweep_design.utility_functions.f_t import f_t_linear_array, f_t_linear_function
from sweep_design.utility_functions.a_t import tukey_a_t


class TestUtilityFunctions(unittest.TestCase):

    def test_freq2time_functions(self):

        freq2time_functions = [proportional_freq2time, dwell(0, 5, 1)]

        for func in freq2time_functions:
            with self.subTest("Test of conversion frequency-time functions", func=func):

                spectrum = Spectrum([0, 1, 2, 3, 4, 5],
                                    [10, 20, 30, 40, 50, 60])
                result = func(spectrum)

                self.assertEqual(len(result), 3)

                self.assertIsInstance(result[0], np.ndarray)
                self.assertIsInstance(result[1], np.ndarray)
                self.assertIsInstance(result[2], np.ndarray)

    def test_emd_analyze(self):

        get_emd_functions = [get_IMFs_emd, get_IMFs_ceemdan]

        for func in get_emd_functions:
            with self.subTest("Test emd function.", func=func):

                signal = Signal([0, 1, 2, 3, 4, 5], [10, 20, 30, 40, 50, 60])

                emd_result = func(signal)

                self.assertGreater(len(emd_result), 0)

    def test_linear_functions(self):
        time_axis = ArrayAxis(0, 10, 0.1)
        func = f_t_linear_function(0, 10, 5, 95)

        self.assertTrue(callable(func))

        func_result = func(time_axis.array)

        self.assertEqual(func_result[0], 5)
        self.assertEqual(func_result[-1], 95)

        array_result = f_t_linear_array(time_axis.array, 10, 75)

        self.assertEqual(array_result[0], 10)
        self.assertEqual(array_result[-1], 75)

    def test_tukey_a_t(self):
        time_axis = ArrayAxis(0, 10, 0.1)

        left_result = tukey_a_t(time_axis.array, 1, "left")

        self.assertEqual(left_result[0], 0)
        self.assertEqual(left_result[-1], 1)

        left_result = tukey_a_t(time_axis.array, 1, "right")

        self.assertEqual(left_result[0], 1)
        self.assertEqual(left_result[-1], 0)

        left_result = tukey_a_t(time_axis.array, 1, "both")

        self.assertEqual(left_result[0], 0)
        self.assertEqual(left_result[-1], 0)
        self.assertEqual(left_result[int(left_result.size / 2)], 1)
