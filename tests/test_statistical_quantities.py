import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from anaspike.functions.statistical_quantities import (pearson_correlation_offset_data,
                                                       radial_average,
                                                       morans_i)



class TestPearsonCorrelationOffsetData(unittest.TestCase):
    def test_error_non_matching_data_shapes(self):
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[1, 2], [3, 4], [5, 6]])
        offset_vectors = np.array([(0, 0)])
        with self.assertRaises(ValueError):
            pearson_correlation_offset_data(data1, data2, offset_vectors)

    def test_error_non_matching_vector_dimensions(self):
        data1 = np.array([[1, 2], [3, 4]])
        data2 = np.array([[1, 2], [3, 4]])
        offset_vectors = np.array([(0, 0, 0),
                                   (0, 0, 1),
                                   (0, 1, 0),
                                   (1, 0, 0)])
        with self.assertRaises(ValueError):
            pearson_correlation_offset_data(data1, data2, offset_vectors)

    def test_1d_correlation(self):
        data1 = np.array([0, 1, 0])
        data2 = np.array([1, 0, -1])
        offset_vectors = np.array([(-1,), (0,), (1,)])
        corr = pearson_correlation_offset_data(data1, data2, offset_vectors)
        expected_corr = [-1, 0, 1]
        np.testing.assert_array_almost_equal(corr, expected_corr)

    def test_1d_correlation_sine(self):
        delta = 0.2
        x = np.arange(0, 2*np.pi, delta)
        static_data = np.sin(x)
        to_offset_data = np.sin(x + 2*delta)
        offset_vectors = np.array([(-1,), (+2,), (0,), (1,)])
        corr = pearson_correlation_offset_data(static_data, to_offset_data, offset_vectors)
        self.assertLess(corr[0], 1.0)
        self.assertAlmostEqual(corr[1], 1.0)
        self.assertLess(corr[2], 1.0)
        self.assertLess(corr[3], 1.0)

    def test_1d_correlation_against_np_corrcoeff(self):
        n = 100
        data1 = np.random.rand(n)
        data2 = np.random.rand(n)
        offset_vectors = np.array([(i,) for i in range(n - 1)])
        corr = pearson_correlation_offset_data(data1, data2, offset_vectors)
        expected_corr = [np.corrcoef(data1, data2)[0, 1]] + [np.corrcoef(data1[i:], data2[:-i])[0, 1] for (i, ) in offset_vectors[1:]]
        np.testing.assert_array_almost_equal(corr, expected_corr)

    def test_2d_correlation_gaussian(self):
        delta = 0.2
        x, y = np.meshgrid(np.arange(-2, 2, delta), np.arange(-2, 2, delta))
        static_data = np.exp(-(x**2 + y**2))
        to_offset_data = np.exp(-((x - delta)**2 + (y + 2*delta)**2))
        offset_vectors = np.array([(-1, -1), (+2, -1), (0, 0), (1, 1)])
        corr = pearson_correlation_offset_data(static_data, to_offset_data, offset_vectors)
        self.assertLess(corr[0], 1.0)
        self.assertAlmostEqual(corr[1], 1.0)
        self.assertLess(corr[2], 1.0)
        self.assertLess(corr[2], 1.0)

    def test_autocorrelation(self):
        data = np.array([[1, 2],
                         [3, 4]])
        offset_vectors = np.array([(0, 0)])
        corr = pearson_correlation_offset_data(data, data, offset_vectors)
        expected_corr = [1.]
        np.testing.assert_array_almost_equal(corr, expected_corr)


class TestRadialAverage(unittest.TestCase):
    def setUp(self):
        """Common mock configuration for Bin class"""
        self.mock_bin_patcher = patch('anaspike.dataclasses.interval.Bin')
        self.mock_bin = self.mock_bin_patcher.start()
        
        self.mock_bin.side_effect = lambda start, end: MagicMock(
            start=start, 
            end=end
        )

    def tearDown(self):
        self.mock_bin_patcher.stop()

    def test_single_point_one_bin(self):
        origin = (0.0, 0.0)
        x = np.array([1.0])
        y = np.array([0.0])
        data = np.array([5.0])
        bins = [self.mock_bin(0.0, 2.0)]
        
        result = radial_average(origin, x, y, data, bins)
        np.testing.assert_array_almost_equal(result, np.array([5.0]))

    def test_multiple_points_one_bin(self):
        origin = (0.0, 0.0)
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        data = np.array([3.0, 7.0])
        bins = [self.mock_bin(0.0, 2.0)]
        
        result = radial_average(origin, x, y, data, bins)
        np.testing.assert_array_almost_equal(result, np.array([5.0]))

    def test_multiple_points_multiple_bins(self):
        origin = (0.0, 0.0)
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.0, 0.0, 0.0])
        data = np.array([2.0, 4.0, 6.0])
        bins = [
            self.mock_bin(0.5, 1.5),
            self.mock_bin(1.5, 2.5),
            self.mock_bin(2.5, 3.5)
        ]
        
        result = radial_average(origin, x, y, data, bins)
        np.testing.assert_array_almost_equal(result, np.array([2.0, 4.0, 6.0]))

    def test_empty_bin(self):
        origin = (0.0, 0.0)
        x = np.array([0.5, 2.5])
        y = np.array([0.0, 0.0])
        data = np.array([10.0, 20.0])
        bins = [
            self.mock_bin(0.0, 1.0),
            self.mock_bin(1.0, 2.0),
            self.mock_bin(2.0, 3.0)
        ]
        
        result = radial_average(origin, x, y, data, bins)
        np.testing.assert_array_almost_equal(result, np.array([10.0, 0.0, 20.0]))

    def test_empty_bins(self):
        origin = (0.0, 0.0)
        x = np.array([])
        y = np.array([])
        data = np.array([])
        bins = [
            self.mock_bin(0.0, 1.0),
            self.mock_bin(1.0, 2.0)
        ]
        
        result = radial_average(origin, x, y, data, bins)
        np.testing.assert_array_almost_equal(result, np.array([0.0, 0.0]))

    def test_origin_shift(self):
        origin = (2.0, 3.0)
        x = np.array([2.0])
        y = np.array([4.0])
        data = np.array([10.0])
        bins = [self.mock_bin(0.5, 1.5)]
        
        result = radial_average(origin, x, y, data, bins)
        np.testing.assert_array_almost_equal(result, np.array([10.0]))

    def test_bin_edge_points(self):
        origin = (0.0, 0.0)
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.0, 0.0, 0.0])
        data = np.array([10.0, 20.0, 30.0])
        bins = [
            self.mock_bin(1.0, 2.0),
            self.mock_bin(2.0, 3.0)
        ]
        
        result = radial_average(origin, x, y, data, bins)
        np.testing.assert_array_almost_equal(result, np.array([10.0, 25.0]))

    def test_point_at_origin(self):
        origin = (0.0, 0.0)
        x = np.array([0.0])
        y = np.array([0.0])
        data = np.array([5.0])
        bins = [self.mock_bin(0.0, 1.0)]
        
        result = radial_average(origin, x, y, data, bins)
        np.testing.assert_array_almost_equal(result, np.array([5.0]))

    def test_non_uniform_bins(self):
        origin = (0.0, 0.0)
        x = np.array([0.5, 1.5, 2.5, 3.5])
        y = np.array([0.0, 0.0, 0.0, 0.0])
        data = np.array([1.0, 2.0, 3.0, 4.0])
        bins = [
            self.mock_bin(0.0, 1.0),
            self.mock_bin(1.0, 3.0),
            self.mock_bin(3.0, 4.0)
        ]
        
        result = radial_average(origin, x, y, data, bins)
        np.testing.assert_array_almost_equal(result, np.array([1.0, 2.5, 4.0]))


class TestMoransI(unittest.TestCase):
    def test_error_non_1d_data(self):
        data = np.array([[1, 2], [3, 4]])
        weights = np.array([[0, 1], [1, 0]])
        with self.assertRaises(ValueError):
            morans_i(data, weights)

    def test_error_incompatible_weights(self):
        data = np.array([1, 2, 3])
        weights = np.array([[0, 1, 0], [0, 1, 0]])
        with self.assertRaises(ValueError):
            morans_i(data, weights)

    def test_chessboard(self):
        chessboard_data = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [1, 0, 1, 0],
                                    [0, 1, 0, 1]], dtype=np.float64).flatten()
        rook_weights = np.array([[0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                 [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                 [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                 [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                                 [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
                                 [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]], dtype=np.float64)
        result = morans_i(chessboard_data, rook_weights)
        expected_result = -1.0
        self.assertAlmostEqual(result, expected_result)

    def test_random(self):
        n = 10
        data = np.random.rand(n)
        weights = np.random.rand(n, n)
        result = morans_i(data, weights)

        diff = data - np.mean(data)
        w = 0.
        numerator = 0.
        denominator = 0.
        for i in range(n):
            denominator += diff[i]**2
            for j in range(n):
                w += weights[i, j]
                numerator += weights[i, j] * diff[i] * diff[j]
        expected_result = (n / w) * (numerator / denominator)

        self.assertAlmostEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main(verbosity=2)

