import unittest

import numpy as np

from sweep_design.axis import get_array_axis_from_array, ArrayAxis
from sweep_design.help_types import Number


class TestMethodsToCreateAxis(unittest.TestCase):

    def check_axis(self, axis: ArrayAxis, x_start: Number, x_end: Number,
                   sample: Number):
        self.assertEqual(axis.start, x_start)
        self.assertEqual(axis.end, x_end)
        self.assertEqual(axis.sample, sample)

    def test_get_axis_from_list(self):

        int_array1 = [2, 3, 4, 5, 6, 7]
        int_axis_1 = get_array_axis_from_array(int_array1)
        self.check_axis(int_axis_1, 2, 7, 1)

        int_array2 = [-5, 0, 5, 10, 15, 20, 25]
        int_axis_2 = get_array_axis_from_array(int_array2)
        self.check_axis(int_axis_2, -5, 25, 5)

        float_array1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        float_axis1 = get_array_axis_from_array(float_array1)
        self.check_axis(float_axis1, 0, 0.5, 0.1)

        float_array2 = [-5.5, -2.5, 1.5, 4.5]
        float_axis2 = get_array_axis_from_array(float_array2)
        self.check_axis(float_axis2, -5.5, 4.5, 3.0)

    def test_get_axis_from_np_array(self):

        dx = 0.2
        x_start = -3.0
        x_end = 5.6
        num_sample = int((x_end - x_start) / dx) + 1
        np_array = np.linspace(-3.0, 5.6, num_sample)
        np_axis = get_array_axis_from_array(np_array)
        self.check_axis(np_axis, x_start, x_end, dx)

    def test_complex_number(self):
        start = 0j
        end = 10j
        complex_axis = ArrayAxis(start=start, end=end, sample=1j)
        self.assertEqual(complex_axis.array[0], start)
        self.assertEqual(complex_axis.array[-1], end)

    def test_changes_array_axis(self):
        array_axis = ArrayAxis(start=0.0, end=10.0, sample=0.1)

        start_array_axis = array_axis.copy()
        start_array_axis.start = 1.0
        self.assertNotEqual(start_array_axis.array.size, array_axis.array.size)

        end_array_axis = array_axis.copy()
        end_array_axis.end = 9.0
        self.assertNotEqual(end_array_axis.array.size, array_axis.array.size)

        sample_array_axis = array_axis.copy()
        sample_array_axis.sample = 2.0
        self.assertNotEqual(end_array_axis.array.size, array_axis.array.size)
