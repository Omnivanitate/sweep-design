import unittest

from sweep_design.defaults.sweep_methods import InterpolateArray
from sweep_design.exc import BadInputError
from sweep_design.relation import Relation
from sweep_design.sweep import Sweep
from sweep_design.uncalculated_sweep import UncalculatedSweep, ApriorUncalculatedSweep
from sweep_design.axis import ArrayAxis


class TestUncalculatedSweep(unittest.TestCase):
    def setUp(self) -> None:
        self.times = [None, ArrayAxis(0., 10., 0.1), [0, 1, 2, 3, 4, 5, 6]]
        self.second_times = [
            None, ArrayAxis(
                0., 10., 0.1), [
                0, 1, 2, 3, 4, 5, 6]]
        self.f_t = self.a_t = [
            [10, 20, 30, 40, 50, 60, 70],
            lambda t: 10 * t + 1,
            Relation([0, 1, 2, 3, 4, 5], [10, 20, 30, 40, 50, 60]),
            InterpolateArray(
                Relation([0, 1, 2, 3, 4, 5], [10, 20, 30, 40, 50, 60]))
        ]

    def check_creation(self, time, a_t, f_t, second_time):

        unsw = UncalculatedSweep(time, f_t, a_t)

        self.assertIsInstance(unsw, UncalculatedSweep)

        sw = unsw(second_time)

        self.assertIsInstance(sw, Sweep)

        if isinstance(second_time, ArrayAxis):
            self.assertEqual(sw.start, second_time.start)
            self.assertEqual(sw.end, second_time.end)
            self.assertEqual(sw.sample, second_time.sample)

        if isinstance(second_time, list):
            self.assertEqual(sw.start, 0)
            self.assertEqual(sw.end, 6)
            self.assertEqual(sw.sample, 1)

    def test_input_data(self):
        for time in self.times:
            for f_t in self.f_t:
                for a_t in self.a_t:
                    for second_time in self.second_times:
                        with self.subTest(f"Next test params:\n"
                                          f"time: {time}\n"
                                          f"f_t: {f_t}\n"
                                          f"a_t: {a_t}\n"
                                          f"second_time: {second_time}\n"
                                          ):
                            if time is None and (isinstance(f_t, list) or isinstance(
                                    a_t, list)) or (time is None and second_time is None):

                                with self.assertRaises(BadInputError):
                                    self.check_creation(
                                        time, f_t, a_t, second_time)

                                continue

                            self.check_creation(time, f_t, a_t, second_time)

    @unittest.skip("use general test")
    def test_input_array_like_data(self):
        time = ArrayAxis(0., 10., 0.1)
        f_t = InterpolateArray(
            Relation([0, 1, 2, 3, 4, 5], [10, 20, 30, 40, 50, 60]))
        a_t = [10, 20, 30, 40, 50, 60, 70]
        second_time = [0, 1, 2, 3, 4, 5, 6]

        unsw = UncalculatedSweep(time, f_t, a_t)

        self.assertIsInstance(unsw, UncalculatedSweep)

        sw = unsw()

        self.assertIsInstance(sw, Sweep)

        self.assertEqual(sw.start, 0)
        self.assertEqual(sw.end, 10)
        self.assertEqual(sw.sample, 0.1)

        sw = unsw(second_time)

        self.assertIsInstance(unsw, UncalculatedSweep)

        self.assertEqual(sw.start, 0)
        self.assertEqual(sw.end, 6)
        self.assertEqual(sw.sample, 1)

        with self.assertRaises(BadInputError):
            UncalculatedSweep(None, [1, 2, 3], [1, 2, 3])
