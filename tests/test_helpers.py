import unittest

import numpy as np

from anaspike.functions._helpers import (construct_offsets,
                                         construct_offset_vectors,
                                         slice_from_vec,
                                         offset_via_slicing,
                                         validate_same_length,
                                         )



class TestConstructOffsets(unittest.TestCase):
    def test_n_less_than_margin(self):
        n = 10
        margin = 20
        with self.assertRaises(ValueError):
            construct_offsets(n, margin)

    def test_n_greater_than_margin(self):
        n = 21
        margin = 20
        result = construct_offsets(n, margin)
        expected = np.array([-1, 0, 1])
        np.testing.assert_array_equal(result, expected)


class TestConstructOffsetVectors(unittest.TestCase):
    def test_n_x_less_than_margin(self):
        n_x = 10
        n_y = 21
        margin = 20
        with self.assertRaises(ValueError):
            construct_offset_vectors(n_x, n_y, margin)

    def test_n_y_less_than_margin(self):
        n_x = 21
        n_y = 10
        margin = 20
        with self.assertRaises(ValueError):
            construct_offset_vectors(n_x, n_y, margin)

    def test_n_x_and_n_y_greater_than_margin(self):
        n_x = 21
        n_y = 21
        margin = 20
        result = construct_offset_vectors(n_x, n_y, margin)
        expected = np.array([(-1, -1), (-1, 0), (-1, 1),
                             (0 , -1), (0 , 0), (0 , 1),
                             (+1, -1), (+1, 0), (+1, 1)])
        np.testing.assert_array_equal(result, expected)


class TestSliceFromVec(unittest.TestCase):
    def test_1d_array_positive_vector(self):
        v = [2]
        arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = arr[slice_from_vec(v)]
        expected = np.array([2, 3, 4, 5, 6, 7, 8, 9])
        np.testing.assert_array_equal(result, expected)

    def test_1d_array_negative_vector(self):
        v = [-2]
        arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = arr[slice_from_vec(v)]
        expected = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(result, expected)

    def test_1d_array_zero_vector(self):
        v = [0]
        arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = arr[slice_from_vec(v)]
        expected = arr
        np.testing.assert_array_equal(result, expected)

    def test_2d_array_positive_vector(self):
        v = [1, 2]
        arr = np.array([[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]])
        result = arr[slice_from_vec(v)]
        expected = np.array([[5],
                             [8]])
        np.testing.assert_array_equal(result, expected)

    def test_2d_array_negative_vector(self):
        v = [-1, -2]
        arr = np.array([[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]])
        result = arr[slice_from_vec(v)]
        expected = np.array([[0],
                             [3]])
        np.testing.assert_array_equal(result, expected)

    def test_2d_array_mixed_vector(self):
        v = [1, -2]
        arr = np.array([[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]])
        result = arr[slice_from_vec(v)]
        expected = np.array([[3],
                             [6]])
        np.testing.assert_array_equal(result, expected)

    def test_2d_array_zero_vector(self):
        v = [0, 0]
        arr = np.array([[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]])
        result = arr[slice_from_vec(v)]
        expected = arr
        np.testing.assert_array_equal(result, expected)


class TestOffsetViaSlicing(unittest.TestCase):
    def test_error_non_matching_data_shapes(self):
        static_data = np.array([[1, 2], [3, 4]])
        to_offset_data = np.array([[1, 2], [3, 4], [5, 6]])
        offset_vector = np.array([0, 0])
        with self.assertRaises(ValueError):
            offset_via_slicing(static_data, to_offset_data, offset_vector)

    def test_error_non_matching_vector_dimensions(self):
        static_data = np.array([[1, 2], [3, 4]])
        to_offset_data = np.array([[1, 2], [3, 4]])
        offset_vector = np.array([0, 0, 0])
        with self.assertRaises(ValueError):
            offset_via_slicing(static_data, to_offset_data, offset_vector)

    def test_1d_offset(self):
        data = np.array([0, 1, 2, 3, 4, 5])
        offset_vector = np.array([1])
        static_slice, offset_slice = offset_via_slicing(data, data, offset_vector)
        product = static_slice * offset_slice
        expected_product = np.array([0, 2, 6, 12, 20])
        np.testing.assert_array_equal(product, expected_product)

    def test_2d_offset(self):
        data = np.array([[0, 1, 2],
                         [3, 4, 5],
                         [6, 7, 8]])
        offset_vector = np.array([1, 1])
        static_slice, offset_slice = offset_via_slicing(data, data, offset_vector)
        product = static_slice * offset_slice
        expected_product = np.array([[0, 5],
                                     [21, 32]])
        np.testing.assert_array_equal(product, expected_product)


class TestValidateSameLength(unittest.TestCase):
    def test_no_args(self):
        validate_same_length()

    def test_one_arg(self):
        validate_same_length([1, 2, 3])

    def test_same_length(self):
        validate_same_length([1, 2, 3], [4, 5, 6], [7, 8, 9])

    def test_different_length(self):
        with self.assertRaises(ValueError):
            validate_same_length([1, 2, 3], [4, 5, 6], [7, 8])

    def test_different_length_2(self):
        with self.assertRaises(ValueError):
            validate_same_length([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11])


if __name__ == '__main__':
    unittest.main(verbosity=2)

