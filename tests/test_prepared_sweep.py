import unittest

from sweep_design.axis import ArrayAxis
from sweep_design.prepared_sweeps.code_m_sequence import (
    get_m_sequence_code, get_relation_m_sequence)
from sweep_design.prepared_sweeps.code_zinger import (get_code_zinger,
                                                      get_code_zinger_relation)
from sweep_design.prepared_sweeps.dwell_sweep import get_dwell_sweep
from sweep_design.prepared_sweeps.linear_sweep import get_linear_sweep
from sweep_design.prepared_sweeps.pseudorandom_shuffle import get_shuffle
from sweep_design.prepared_sweeps.sweep_from_code import (
    get_code_sweep_segments, get_convolution_sweep_and_code)
from sweep_design.relation import Relation
from sweep_design.sweep import Sweep


class TestPreparedSweep(unittest.TestCase):

    def test_linear_sweep(self):
        time = ArrayAxis(0., 10., 0.01)

        linear_sweep = get_linear_sweep(time, 5, 95, None)
        self.assertIsInstance(linear_sweep, Sweep)

        linear_sweep = get_linear_sweep(time, 5, 95, 1)
        self.assertIsInstance(linear_sweep, Sweep)

    def test_dwell_sweep(self):
        time = ArrayAxis(0., 10., 0.01)
        aprior_data = get_linear_sweep(time, 5, 95, None)

        result_sweep = get_dwell_sweep(time.array, 6, 10, 90, 1, aprior_data)
        self.assertIsInstance(result_sweep, Sweep)

    def test_shuffle_sweep(self):
        time = ArrayAxis(0., 10., 0.01)
        shuffle_sweep = get_shuffle(time, 5, 100, 0.5, time_tapper=2)
        self.assertIsInstance(shuffle_sweep, Sweep)

    def test_m_sequence_code_sweep(self):
        m_sequence = get_m_sequence_code(5)
        m_sequence_full = get_m_sequence_code(5, is_full=True)

        m_sequence_time = ArrayAxis(0, 0.05, 0.01)
        m_relation = get_relation_m_sequence(m_sequence_time)
        m_relation_full = get_relation_m_sequence(
            m_sequence_time, is_full=True)

        time = ArrayAxis(0, 10, 0.01)
        linear_sweep = get_linear_sweep(time, 5, 95, 1)

        segment_sweep_relation = get_code_sweep_segments(
            m_relation, linear_sweep)
        self.assertIsInstance(segment_sweep_relation, Sweep)

        segment_sweep_relation = get_code_sweep_segments(
            m_relation_full, linear_sweep)
        self.assertIsInstance(segment_sweep_relation, Sweep)

        segment_sweep_relation = get_code_sweep_segments(
            m_sequence_full, linear_sweep)
        self.assertIsInstance(segment_sweep_relation, Sweep)

        corr_sweep_relation = get_convolution_sweep_and_code(
            m_sequence, linear_sweep)
        self.assertIsInstance(corr_sweep_relation, Sweep)

    def test_code_zinger(self):

        code_zinger = get_code_zinger([-1, -1, -1, 1])

        zinger_code_time = ArrayAxis(0, 0.03, 0.01)
        code_zinger_relation = \
            get_code_zinger_relation(
                Relation(
                    zinger_code_time, [-1, -1, -1, 1]
                ), periods=2)

        time = ArrayAxis(0, 10, 0.01)
        linear_sweep = get_linear_sweep(time, 5, 95, 1)

        segment_sweep_relation = get_code_sweep_segments(
            code_zinger_relation, linear_sweep)
        self.assertIsInstance(segment_sweep_relation, Sweep)

        corr_sweep_relation = get_convolution_sweep_and_code(
            code_zinger_relation, linear_sweep)
        self.assertIsInstance(corr_sweep_relation, Sweep)

        segment_sweep_array = get_code_sweep_segments(
            code_zinger, linear_sweep)
        self.assertIsInstance(segment_sweep_array, Sweep)

        corr_sweep_array = get_convolution_sweep_and_code(
            code_zinger, linear_sweep)
        self.assertIsInstance(corr_sweep_array, Sweep)
