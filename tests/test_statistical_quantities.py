import unittest

import numpy as np

from anaspike.functions import pearson_correlation_offset_data



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



if __name__ == '__main__':
    unittest.main(verbosity=2)

