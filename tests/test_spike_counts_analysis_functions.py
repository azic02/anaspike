import unittest

import numpy as np

from anaspike.analysis.spike_counts import (SpikeCounts,
                                            get_active_neurons_fraction)
from anaspike.analysis.spike_trains import SpikeTrains
from anaspike.dataclasses.interval import Interval
from anaspike.dataclasses.coords2d import Coords2D



class CalculateActiveNeuronFractionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t_interval = Interval(0, 2)
        cls.thresh = 1

class TestCalculateActiveNeuronFractionSomeActive(CalculateActiveNeuronFractionTestCase):
    def setUp(self):
        self.coords = Coords2D(np.arange(4), np.arange(4))
        self.spike_trains = SpikeTrains(self.coords, [np.array([0, 0.1]),
                                                      np.array([1.0, 1.2]),
                                                      np.array([0.3]),
                                                      np.array([])])

        self.expected_fraction = 3. / 4

    def test(self):
        sc = SpikeCounts.from_spike_trains(self.spike_trains, self.t_interval)
        result = get_active_neurons_fraction(sc, self.thresh)
        self.assertAlmostEqual(result, self.expected_fraction)

class TestCalculateActiveNeuronFractionNoneActive(CalculateActiveNeuronFractionTestCase):
    def setUp(self):
        self.coords = Coords2D(np.arange(3), np.arange(3))
        self.spike_trains = SpikeTrains(self.coords, [np.array([]),
                                                      np.array([]),
                                                      np.array([])])

        self.expected_fraction = 0.0

    def test(self):
        sc = SpikeCounts.from_spike_trains(self.spike_trains, self.t_interval)
        result = get_active_neurons_fraction(sc, self.thresh)
        self.assertAlmostEqual(result, self.expected_fraction)

class TestCalculateActiveNeuronFractionAllActive(CalculateActiveNeuronFractionTestCase):
    def setUp(self):
        self.coords = Coords2D(np.arange(3), np.arange(3))
        self.spike_trains = SpikeTrains(self.coords, [np.array([0]),
                                                      np.array([1.0]),
                                                      np.array([0.3])])

        self.expected_fraction = 1.0

    def test(self):
        sc = SpikeCounts.from_spike_trains(self.spike_trains, self.t_interval)
        result = get_active_neurons_fraction(sc, self.thresh)
        self.assertAlmostEqual(result, self.expected_fraction)
