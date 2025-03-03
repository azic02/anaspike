import unittest

import numpy as np

from anaspike.dataclasses.nest_devices import PopulationData



class SharedSetup(unittest.TestCase):
    def setUp(self):
        self.population_data = PopulationData(
            np.array([1, 2, 3], dtype=np.int64),
            np.array([0.5, 1.0, 1.5], dtype=np.float64),
            np.array([1.5, 2.0, 2.5], dtype=np.float64),
        )


class TestGetItem(SharedSetup):
    def test_single_neuron(self):
        neuron = self.population_data[1]

        expected_ids = np.array([2], dtype=np.int64)
        expected_x_pos = np.array([1.0], dtype=np.float64)
        expected_y_pos = np.array([2.0], dtype=np.float64)

        np.testing.assert_array_equal(neuron.ids, expected_ids)
        np.testing.assert_array_almost_equal(neuron.x_pos, expected_x_pos)
        np.testing.assert_array_almost_equal(neuron.y_pos, expected_y_pos)

    def test_multiple_neurons(self):
        neurons = self.population_data[[0, 2]]

        expected_ids = np.array([1, 3], dtype=np.int64)
        expected_x_pos = np.array([0.5, 1.5], dtype=np.float64)
        expected_y_pos = np.array([1.5, 2.5], dtype=np.float64)

        np.testing.assert_array_equal(neurons.ids, expected_ids)
        np.testing.assert_array_almost_equal(neurons.x_pos, expected_x_pos)
        np.testing.assert_array_almost_equal(neurons.y_pos, expected_y_pos)

    def test_sliced_neurons(self):
        neurons = self.population_data[1:]

        expected_ids = np.array([2, 3], dtype=np.int64)
        expected_x_pos = np.array([1.0, 1.5], dtype=np.float64)
        expected_y_pos = np.array([2.0, 2.5], dtype=np.float64)

        np.testing.assert_array_equal(neurons.ids, expected_ids)
        np.testing.assert_array_almost_equal(neurons.x_pos, expected_x_pos)
        np.testing.assert_array_almost_equal(neurons.y_pos, expected_y_pos)


    def test_masked_neurons(self):
        mask = np.array([True, False, True])
        neurons = self.population_data[mask]

        expected_ids = np.array([1, 3], dtype=np.int64)
        expected_x_pos = np.array([0.5, 1.5], dtype=np.float64)
        expected_y_pos = np.array([1.5, 2.5], dtype=np.float64)

        np.testing.assert_array_equal(neurons.ids, expected_ids)
        np.testing.assert_array_almost_equal(neurons.x_pos, expected_x_pos)
        np.testing.assert_array_almost_equal(neurons.y_pos, expected_y_pos)


class TestAdd(SharedSetup):
    def setUp(self):
        super().setUp()
        self.population_data_2 = PopulationData(
            np.array([4, 5], dtype=np.int64),
            np.array([16., 17.], dtype=np.float64),
            np.array([26., 27.], dtype=np.float64),
        )

    def test_add_single(self):
        result = self.population_data + self.population_data_2

        expected_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        expected_x_pos = np.array([0.5, 1.0, 1.5, 16., 17.], dtype=np.float64)
        expected_y_pos = np.array([1.5, 2.0, 2.5, 26., 27.], dtype=np.float64)

        np.testing.assert_array_equal(result.ids, expected_ids)
        np.testing.assert_array_almost_equal(result.x_pos, expected_x_pos)
        np.testing.assert_array_almost_equal(result.y_pos, expected_y_pos)

