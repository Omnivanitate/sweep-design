from typing import Union
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from sweep_design.relation import Relation
from sweep_design.axis import ArrayAxis
from sweep_design.help_types import Number
from sweep_design.exc import NotEqualError, BadInputError, TypeFuncError


class WrapperTestRelation:

    class TestBaseRelation(unittest.TestCase):

        def setUp(self) -> None:
            self.relation_class = Relation
            self.x_axis = ArrayAxis(start=0, end=0.5, sample=0.1)
            self.relation = self.relation_class(self.x_axis,
                                                [10.1, -5.231, 123., 0., 12.465, 5.])

            self.simple_relation = self.relation_class(self.x_axis,
                                                       [10, 20, 30, 40, 50, 60])

        def tearDown(self) -> None:
            pass

        def test_init_relation(self):
            start = 0.0
            end = 10.0
            sample = 0.05

            x_array_axis = ArrayAxis(start=start, end=end, sample=sample)
            y = np.ones(x_array_axis.array.size)
            relation_array_axis = self.relation_class(x_array_axis, y)
            self.assertAlmostEqual(sample, relation_array_axis.actual_sample)

            x = np.linspace(start, end, int((end - start) / sample) + 1)
            relation_from_np = self.relation_class(x, y)
            self.assertAlmostEqual(sample, relation_from_np.actual_sample)

            relation_from_relation = self.relation_class(relation_from_np)
            self.assertAlmostEqual(
                sample, relation_from_relation.actual_sample)

            relation_array_like = self.relation_class(
                [0, 1, 2, 3, 4, 5], [10, 20, 30, 40, 50, 60])
            self.assertAlmostEqual(1, relation_array_like.actual_sample)

            r = self.relation_class(
                [0.1, 0.2, 0.3], [2.123123, 10.123, -10.123])
            self.assertAlmostEqual(1, relation_array_like.actual_sample)

            with self.assertRaises(NotEqualError):
                self.relation_class([1, 2, 3], [1, 2])

            with self.assertRaises(BadInputError):
                self.relation_class([1, 2, 3], None)

        def test_properties(self):
            self.assertIsInstance(self.relation.x, ArrayAxis)

            relation_array_like = self.relation_class([1, 2, 3], [1, 2, 3])
            self.assertIsInstance(relation_array_like.x, ArrayAxis)
            self.assertEqual(relation_array_like.start, 1)
            self.assertEqual(relation_array_like.end, 3)
            self.assertEqual(relation_array_like.sample, 1)
            assert_array_equal([1, 2, 3], relation_array_like.array)

            self.assertIs(self.relation.x, self.x_axis)
            self.assertIsInstance(self.relation.sample, (int, float, complex))
            self.assertEqual(self.relation.sample, 0.1)

            self.assertIs(self.relation.x, self.x_axis)

            self.relation.start = 5
            self.assertEqual(self.relation.start, 5)

            self.relation.end = 11
            self.assertEqual(self.relation.end, 11)

            self.relation.sample = 1
            self.assertEqual(self.relation.sample, 1)

            assert_array_equal(self.relation.array, [5, 6, 7, 8, 9, 10, 11])

            with self.assertRaises(NotEqualError):
                self.relation.get_data()

        def test_max(self):
            r = self.relation_class(
                [0.1, 0.2, 0.3], [2.123123, 10.123, -10.123])
            self.assertEqual(r.max(), 10.123)

        def test_min(self):
            r = self.relation_class([0.0, 0.1, 0.2, 0.3], [
                0.0, 2.123123, 10.123, -10.123])
            self.assertEqual(r.min(), -10.123)

        def test_get_norm(self):
            sample_rate = self.relation.get_norm()
            self.assertIsInstance(sample_rate, float)

        def test_get_data(self):
            x, y = self.relation.get_data()

            assert_array_almost_equal(x, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            assert_array_almost_equal(y, [10.1, -5.231, 123., 0., 12.465, 5.])

        def test_select_data(self):
            new_relation = self.relation.select_data(0.1, 0.45)
            self.assertIsNot(self.relation, new_relation)
            self.assertEqual(new_relation.start, 0.1)
            self.assertEqual(new_relation.end, 0.4)
            self.assertEqual(new_relation.sample, 0.1)

            selected_number = self.simple_relation[0.3]
            self.assertAlmostEqual(selected_number[0], 0.3)
            self.assertAlmostEqual(selected_number[1], 40)

            selected_number = self.simple_relation[0.35]
            self.assertAlmostEqual(selected_number[0], 0.3)
            self.assertAlmostEqual(selected_number[1], 40)

            selected_number = self.simple_relation[0.25]
            self.assertAlmostEqual(selected_number[0], 0.2)
            self.assertAlmostEqual(selected_number[1], 30)

            selected_relation = self.simple_relation[0.25:0.43]
            self.assertAlmostEqual(selected_relation.start, 0.3)
            self.assertAlmostEqual(selected_relation.end, 0.4)
            self.assertAlmostEqual(selected_relation.sample, 0.1)

            assert_array_almost_equal(selected_relation.y, [40, 50])

        def test_expanent(self):
            x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            y = [10, 20, 30, 40, 50, 60]

            relation = self.relation_class(x, y)

            exp_relation = relation.exp()
            new_x, new_y = exp_relation.get_data()

            assert_array_almost_equal(x, new_x)
            assert_array_equal(np.exp(y), new_y)

        def test_differentiation(self):

            diff_relation = self.simple_relation.diff()

            self.assertIsInstance(diff_relation, self.relation_class)

            diff_x, diff_y = diff_relation.get_data()

            assert_array_equal(diff_x,
                               self.x_axis.array[:-1] + self.x_axis.sample / 2)

            assert_array_equal(diff_y,
                               [100, 100, 100, 100, 100])

        def test_integration(self):
            integrate_relation = self.simple_relation.integrate()

            self.assertIsInstance(integrate_relation, self.relation_class)

            integer_x, integer_y = integrate_relation.get_data()

            assert_array_equal(integer_x,
                               self.x_axis.array[1:])

            assert_array_equal(integer_y,
                               [1.5, 4., 7.5, 12., 17.5])

        def test_interpolate_extrapolate(self):

            new_x = ArrayAxis(0.0, 0.5, 0.05)

            new_r = self.simple_relation.interpolate_extrapolate(new_x)

            new_rx, new_ry = new_r.get_data()

            assert_array_equal(new_rx, new_x.array)
            assert_array_equal(
                new_ry, [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
            )

            new_x = ArrayAxis(-0.1, 0.6, 0.05)

            new_r = self.simple_relation.interpolate_extrapolate(new_x)

            new_rx, new_ry = new_r.get_data()

            assert_array_equal(new_rx, new_x.array)
            assert_array_almost_equal(
                new_ry, [0., 0., 0., 15., 20., 25., 30., 35.,
                         40., 45., 50., 55., 60., 0., 0.])

        def test_shift(self):

            shift_r = self.simple_relation.shift(0.05)

            self.assertEqual(shift_r.start, 0.05)
            self.assertEqual(shift_r.end, 0.55)
            self.assertEqual(shift_r.sample, 0.1)

            shift_x, shift_y = shift_r.get_data()

            assert_array_equal(shift_x, self.x_axis.array + 0.05)
            assert_array_equal(shift_y, [10, 20, 30, 40, 50, 60])

        def test_equalize(self):

            relation = self.relation_class(
                [-0.1, -0.05, 0.0, 0.05, 0.1, 0.15],
                [1., 10., -10., 2.5, -2.5, 1., ])

            new_relation, new_simple_relation = self.relation_class.equalize(
                relation, self.simple_relation)

            self.assertEqual(new_relation.sample, 0.05)
            self.assertEqual(new_simple_relation.sample, 0.05)

            self.assertEqual(new_relation.start, -0.1)
            self.assertEqual(new_simple_relation.start, -0.1)

            self.assertEqual(new_relation.end, 0.5)
            self.assertEqual(new_simple_relation.end, 0.5)

            assert_array_almost_equal(
                new_relation.y,
                [1., 10., -10., 2.5, -2.5, 1., 0., 0., 0., 0., 0., 0., 0.]
            )

            assert_array_almost_equal(
                new_simple_relation.y,
                [0., 0., 0., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.]
            )

        def test_correlate(self):
            one_relation = self.relation_class(
                self.x_axis,
                np.zeros(self.x_axis.size)
            )
            one_relation.y[0] = 1.
            result = self.relation_class.correlate(
                self.simple_relation, one_relation)

            self.assertIsInstance(result, self.relation_class)

        def test_convolve(self):

            one_relation = self.relation_class(
                self.x_axis,
                np.zeros(self.x_axis.size)
            )
            one_relation.y[0] = 1.

            result = self.relation_class.convolve(
                self.simple_relation, one_relation)

            self.assertIsInstance(result, self.relation_class)

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

        @staticmethod
        def _math_check(
                r1: Relation, ry: Union[Relation, Number], x: ArrayAxis,
                y1: np.array, y: Union[np.ndarray, Number], operation: str):
            r: Relation = r1.__getattribute__(operation)(ry)
            if operation != "__pow__":
                y = y1.__getattribute__(operation)(y)
            else:
                if isinstance(y1, np.ndarray):
                    y = np.abs(y1).__getattribute__(operation)(y) * np.sign(y1)
                else:
                    y = y1.__getattribute__(operation)(y)

            result_x, result_y = r.get_data()

            assert_array_equal(result_x, x.array)
            assert_array_equal(result_y, y)


class TestRelation(WrapperTestRelation.TestBaseRelation):
    pass
