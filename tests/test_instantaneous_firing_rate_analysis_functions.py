import unittest
from typing import cast

import numpy as np

from anaspike.analysis.instantaneous_firing_rate import (InstantaneousFiringRates, 
                                                         temporal_correlation,
                                                         )
from anaspike.dataclasses.nest_devices.population_data import PopulationData, NeuronData
from anaspike.dataclasses.nest_devices.spike_recorder_data import SpikeRecorderData
from anaspike.dataclasses.interval import Bin
from anaspike.analysis.deprecated_functions import (firing_rate_temporal_correlation,
                                                   )


class CommonSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.population_data = PopulationData(
            ids=np.array([0, 1, 2]),
            x_pos=np.array([0.1, 2.0, 0.25]),
            y_pos=np.array([0.5, 2.5, 0.75])
        )

        cls.spike_recorder_data = SpikeRecorderData(
                senders=np.array([0, 0, 2, 1, 1, 1], dtype=np.int64),
                times=np.array([0.1, 0.5, 0.75, 1.1, 2.1, 2.5], dtype=np.float64)
                )

        cls.t_bins = [Bin(start=float(i), end=float(i + 1), value=float(i + 0.5)) for i in range(3)]

        cls.times = np.array([t_bin.value for t_bin in cls.t_bins])

        cls.instantaneous_firing_rates = InstantaneousFiringRates(
            times=cls.times,
            firing_rates=np.array([[2000.0, 0000.0, 0000.0],
                                   [0000.0, 1000.0, 2000.0],
                                   [1000.0, 0000.0, 0000.0]])
            )

        cls.x_bins = [Bin(start=float(i), end=float(i + 1), value=float(i + 0.5)) for i in range(2)]
        cls.y_bins = [Bin(start=float(i), end=float(i + 1), value=float(i + 0.5)) for i in range(3)]


class TestTemporalCorrelation(CommonSetup):
    def test_deprecated_vs_new_implementation(self):
        ref_neuron_idx = 0
        ref_neuron_data = cast(NeuronData, self.population_data[ref_neuron_idx])
        deprecated_result = firing_rate_temporal_correlation(ref_neuron_data,
                                                             self.population_data,
                                                             self.spike_recorder_data,
                                                             self.t_bins)
        ref_firing_rate = self.instantaneous_firing_rates.along_neuron_dim[ref_neuron_idx]
        new_result = temporal_correlation(self.instantaneous_firing_rates, ref_firing_rate)
        np.testing.assert_array_almost_equal(new_result, deprecated_result)

    def test_non_matching_time_dim(self):
        ref_firing_rate = np.array([0., 1., 2., 3.])
        with self.assertRaises(ValueError):
            temporal_correlation(self.instantaneous_firing_rates, ref_firing_rate)

    def test_invalid_reference_firing_rate_shape(self):
        ref_firing_rate = np.array([[0., 1.], [2., 3.]])
        with self.assertRaises(ValueError):
            temporal_correlation(self.instantaneous_firing_rates, ref_firing_rate)


class TestTemporalCorrelationMatrix(CommonSetup):
    def test_deprecated_vs_new_implementation(self):
        from anaspike.analysis.instantaneous_firing_rate import temporal_correlation_matrix
        from anaspike.analysis.deprecated_functions import pairwise_temporal_correlation_matrix as deprecated_pairwise

        deprecated_result = deprecated_pairwise(self.population_data, self.spike_recorder_data, self.t_bins)
        new_result = temporal_correlation_matrix(self.instantaneous_firing_rates)
        np.testing.assert_array_almost_equal(new_result, deprecated_result)


class TestMoransIEvolution(CommonSetup):
    def test_deprecated_vs_new_implementation(self):
        from anaspike.analysis.deprecated_functions import morans_i_evolution as deprecated_morans_i
        from anaspike.analysis.instantaneous_firing_rate import morans_i_evolution
        from anaspike.dataclasses.coords2d import Coords2D
        coords = Coords2D(self.population_data.x_pos, self.population_data.y_pos)
        decay_power = 1.0
        deprecated_result = deprecated_morans_i(self.population_data, 
                                                self.spike_recorder_data,
                                                self.t_bins,
                                                decay_power)
        new_result = morans_i_evolution(self.instantaneous_firing_rates, coords, decay_power)
        np.testing.assert_array_almost_equal(new_result, deprecated_result)


if __name__ == "__main__":
    unittest.main(verbosity=2)

