import unittest

import numpy as np

from anaspike.dataclasses.contig_bins_2d import (ContigBins2D,
                                                 assign_bins,
                                                 calculate_bin_counts,
                                                 calculate_bin_sums,
                                                 calculate_bin_means)
from anaspike.dataclasses.histogram import ContigBins
from anaspike.dataclasses.coords2d import Coords2D



class TestClassSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bins = ContigBins2D(x=ContigBins(edges=np.array([0, 1, 2, 3]),
                                             values=np.array([0.5, 1.5, 2.5])),
                                y=ContigBins(edges=np.array([0, 1, 2]),
                                             values=np.array([0.5, 1.5])))
        cls.coords = Coords2D(x=[0.5, 0.5, 2.5, 0.5],
                              y=[0.5, 1.5, 1.5, 1.5])
        cls.coords_outside_bins = Coords2D(x=[0.5, 0.5, 2.5, 3.5],
                                           y=[0.5, 1.5, 1.5, 1.5])

class TestAssignBins(TestClassSetup):
    def test_coords_outside_bins(self):
        with self.assertRaises(ValueError):
            assign_bins(self.bins, self.coords_outside_bins)

    def test_assign_bins(self):
        expected_x_idxs = np.array([0, 0, 2, 0])
        expected_y_idxs = np.array([0, 1, 1, 1])
        x_idxs, y_idxs = assign_bins(self.bins, self.coords)
        np.testing.assert_array_equal(x_idxs, expected_x_idxs)
        np.testing.assert_array_equal(y_idxs, expected_y_idxs)

class TestCalculateBinCounts(TestClassSetup):
    def test_coords_outside_bins(self):
        with self.assertRaises(ValueError):
            calculate_bin_counts(self.bins, self.coords_outside_bins)

    def test_calculate_bin_counts(self):
        expected_counts = np.array([[1, 2],
                                    [0, 0],
                                    [0, 1]])
        bin_counts = calculate_bin_counts(self.bins, self.coords)
        np.testing.assert_array_equal(bin_counts, expected_counts)

class TestCalculateBinSums(TestClassSetup):
    def test_coords_outside_bins(self):
        arr = np.array(np.arange(4) + 1, dtype=np.float64)
        with self.assertRaises(ValueError):
            calculate_bin_sums(self.bins, self.coords_outside_bins, arr)

    def test_arr_coords_length_mismatch(self):
        arr = np.array(np.arange(3) + 1, dtype=np.float64)
        with self.assertRaises(ValueError):
            calculate_bin_sums(self.bins, self.coords, arr)

    def test_calculate_bin_sums_1d(self):
        arr = np.array(np.arange(4) + 1, dtype=np.float64)
        expected_sums = np.array([[1, 6],
                                  [0, 0],
                                  [0, 3]])
        bin_sums = calculate_bin_sums(self.bins, self.coords, arr)
        np.testing.assert_array_almost_equal(bin_sums, expected_sums)

    def test_calculate_bin_sums_2d(self):
        arr = np.array([[1, 2],
                        [3, 4],
                        [5, 6],
                        [7, 8]])
        expected_sums = np.array([[[1, 2], [10, 12]],
                                  [[0, 0], [0, 0]],
                                  [[0, 0], [5, 6]]
                                 ])
        bin_sums = calculate_bin_sums(self.bins, self.coords, arr)
        np.testing.assert_array_almost_equal(bin_sums, expected_sums)

class TestCalculateBinMeans(TestClassSetup):
    def test_coords_outside_bins(self):
        arr = np.array(np.arange(4) + 1, dtype=np.float64)
        with self.assertRaises(ValueError):
            calculate_bin_means(self.bins, self.coords_outside_bins, arr)

    def test_arr_coords_length_mismatch(self):
        arr = np.array(np.arange(3) + 1, dtype=np.float64)
        with self.assertRaises(ValueError):
            calculate_bin_means(self.bins, self.coords, arr)

    def test_calculate_bin_means_1d(self):
        arr = np.array(np.arange(4) + 1, dtype=np.float64)
        expected_means = np.array([[1 / 1, 6 / 2],
                                   [np.nan, np.nan],
                                   [np.nan, 3 / 1]])
        bin_means = calculate_bin_means(self.bins, self.coords, arr)
        np.testing.assert_array_almost_equal(bin_means, expected_means)

    def test_calculate_bin_means_2d(self):
        arr = np.array([[1, 2],
                        [3, 4],
                        [5, 6],
                        [7, 8]])
        expected_means = np.array([[[1 / 1, 2 / 1],  [10 / 2, 12 / 2]],
                                   [[np.nan, np.nan], [np.nan, np.nan]],
                                   [[np.nan, np.nan], [5 / 1, 6 / 1]]])
        bin_means = calculate_bin_means(self.bins, self.coords, arr)
        np.testing.assert_array_almost_equal(bin_means, expected_means)

