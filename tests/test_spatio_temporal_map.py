import unittest

import numpy as np

from anaspike.dataclasses.spatio_temporal_map import SpatioTemporalMap
from anaspike.dataclasses.coords2d import Coords2D



class TestSpatioTemporalMapInitialisation(unittest.TestCase):
    def test_invalid_time_dimension(self):
        with self.assertRaises(ValueError):
            SpatioTemporalMap(coords=Coords2D(x=[0, 1], y=[0, 1]),
                              times=np.random.rand(3, 2),
                              values=np.random.rand(2, 2, 3))

    def test_invalid_values_dimension(self):
        with self.assertRaises(ValueError):
            SpatioTemporalMap(coords=Coords2D(x=[0, 1], y=[0, 1]),
                              times=np.random.rand(3),
                              values=np.random.rand(2))

    def test_mismatched_times_and_values(self):
        with self.assertRaises(ValueError):
            SpatioTemporalMap(coords=Coords2D(x=[0, 1], y=[0, 1]),
                              times=np.random.rand(4),
                              values=np.random.rand(2, 2))

    def test_mismatched_coords_and_values(self):
        with self.assertRaises(ValueError):
            SpatioTemporalMap(coords=Coords2D(x=[0, 1, 2], y=[0, 1, 3]),
                              times=np.random.rand(3),
                              values=np.random.rand(2, 2))


class TestScalarSpatioTemporalMapProperties(unittest.TestCase):
    def setUp(self):
        self.coords = Coords2D(x=[0, 1], y=[0, 1])
        self.times = np.array([0.0, 1.0, 2.0])
        self.values = np.array([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0]])
        self.map = SpatioTemporalMap(coords=self.coords,
                                     times=self.times,
                                     values=self.values)

    def test_ndim(self):
        self.assertEqual(self.map.ndim, 0)

    def test_shape(self):
        self.assertEqual(self.map.shape, ())


class Test2DVectorSpatioTemporalMapProperties(unittest.TestCase):
    def setUp(self):
        self.coords = Coords2D(x=[0, 1], y=[0, 1])
        self.times = np.array([0.0, 1.0, 2.0])
        self.values = np.arange(120).reshape(2, 3, 4, 5)
        self.map = SpatioTemporalMap(coords=self.coords,
                                     times=self.times,
                                     values=self.values)

    def test_ndim(self):
        self.assertEqual(self.map.ndim, 2)

    def test_shape(self):
        self.assertEqual(self.map.shape, (4, 5))

