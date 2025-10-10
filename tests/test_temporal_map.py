import unittest

import numpy as np
from numpy.testing import assert_array_equal

from anaspike.dataclasses.temporal_map import TemporalMap



class TestTemporalMapInitialisation(unittest.TestCase):
    def test_invalid_time_dimension(self):
        with self.assertRaises(ValueError):
            TemporalMap(times=np.array([[0.0, 1.0], [2.0, 3.0]]),
                        values=np.array([0.0, 1.0]))

    def test_invalid_element_dimension(self):
        with self.assertRaises(ValueError):
            TemporalMap(times=np.array([0.0, 1.0, 2.0]),
                        values=np.array(5.0))

    def test_mismatched_time_and_element_length(self):
        with self.assertRaises(ValueError):
            TemporalMap(times=np.array([0.0, 1.0, 2.0]),
                        values=np.array([0.0, 1.0]))

    def test_mistaken_element_shape(self):
        with self.assertRaises(ValueError):
            TemporalMap(times=np.array([0.0, 1.0, 2.0]),
                        values=np.array([[0.0, 1.0, 2.0],
                                           [3.0, 4.0, 5.0]]))

class TestTemporalMapProperties(unittest.TestCase):
    def test_ndim_1d(self):
        tm = TemporalMap(times=np.array([0.0, 1.0, 2.0]),
                         values=np.array([0.0, 1.0, 2.0]))
        self.assertEqual(tm.ndim, 0)

    def test_ndim_2d(self):
        tm = TemporalMap(times=np.array([0.0, 1.0, 2.0]),
                         values=np.array([[0.0, 1.0],
                                            [2.0, 3.0],
                                            [4.0, 5.0]]))
        self.assertEqual(tm.ndim, 1)

    def test_shape_1d(self):
        tm = TemporalMap(times=np.array([0.0, 1.0, 2.0]),
                         values=np.array([0.0, 1.0, 2.0]))
        assert_array_equal(tm.shape, ())

    def test_shape_2d(self):
        tm = TemporalMap(times=np.array([0.0, 1.0, 2.0]),
                         values=np.array([[0.0, 1.0],
                                            [2.0, 3.0],
                                            [4.0, 5.0]]))
        assert_array_equal(tm.shape, (2,))

    def test_shape_3d(self):
        tm = TemporalMap(times=np.array([0.0, 1.0, 2.0]),
                         values=np.arange(12).reshape(3, 2, 2))
        assert_array_equal(tm.shape, (2, 2))

