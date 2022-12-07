
from sweep_design.axis import ArrayAxis

from sweep_design.sweep import Sweep
from .test_signals import WrapperTestSignal


class TestSweep(WrapperTestSignal.BaseTestSignal):
    def setUp(self) -> None:
        self.relation_class = Sweep
        self.x_axis = ArrayAxis(start=0, end=0.5, sample=0.1)
        self.relation = self.relation_class(self.x_axis,
                                            [10.1, -5.231, 123., 0., 12.465, 5.])

        self.simple_relation = self.relation_class(self.x_axis,
                                                   [10, 20, 30, 40, 50, 60])

        self.x_axis_2 = ArrayAxis(start=0, end=0.6, sample=0.1)
        self.simple_second_relation = self.relation_class(self.x_axis_2,
                                                          [10, 20, 30, 40, 50, 60, 70])
