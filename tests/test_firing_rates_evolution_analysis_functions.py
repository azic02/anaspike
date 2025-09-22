import unittest

import numpy as np

from anaspike.firing_rates_evolution import (FiringRatesEvolution, 
                                             temporal_correlation,
                                             )
from anaspike.dataclasses.interval import Bin



class TemporalCorrelationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t_bins = [Bin(start=float(i), end=float(i + 1), label=float(i + 0.5)) for i in range(3)]

        cls.times = np.array([t_bin.label for t_bin in cls.t_bins])

        cls.firing_rates_evolutions = FiringRatesEvolution(
            times=cls.times,
            firing_rates=np.array([[2000.0, 0000.0, 0000.0],
                                   [0000.0, 1000.0, 2000.0],
                                   [1000.0, 0000.0, 0000.0]])
            )


class TestTemporalCorrelationNonMatchingTimeDim(TemporalCorrelationTestCase):
    def setUp(self):
        self.ref_firing_rate = np.array([0., 1., 2., 3.])
    def test(self):
        with self.assertRaises(ValueError):
            temporal_correlation(self.firing_rates_evolutions, self.ref_firing_rate)

class TestTemporalCorrelationInvalidReferenceShape(TemporalCorrelationTestCase):
    def setUp(self):
        self.ref_firing_rate = np.array([[0., 1.], [2., 3.]])
    def test(self):
        with self.assertRaises(ValueError):
            temporal_correlation(self.firing_rates_evolutions, self.ref_firing_rate)


if __name__ == "__main__":
    unittest.main(verbosity=2)

