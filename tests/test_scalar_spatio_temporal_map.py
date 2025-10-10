import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from anaspike.dataclasses.scalar_spatio_temporal_map import (ScalarSpatioTemporalMap,
                                                             temporal_correlation,
                                                             temporal_correlation_matrix,
                                                             )
from anaspike.dataclasses.coords2d import Coords2D



class TestInitialisation(unittest.TestCase):
    def test_invalid_value_dimension(self):
        with self.assertRaises(ValueError):
            ScalarSpatioTemporalMap(coords=Coords2D(x=[0, 1], y=[0, 1]),
                                    times=np.random.rand(3),
                                    values=np.random.rand(2, 2, 3))


class TestTemporalCorrelation(unittest.TestCase):
    def setUp(self):
        from anaspike.dataclasses.scalar_temporal_map import ScalarTemporalMap
        self.stm = ScalarSpatioTemporalMap(coords=Coords2D(x=np.random.rand(2),
                                                           y=np.random.rand(2)),
                                           times=np.random.random(3),
                                           values=np.array([[1, 2, 3],
                                                            [9, 8, 7]]))

        self.ref = ScalarTemporalMap(times=np.random.rand(3),
                                     values=np.array([4, 5, 6]))

        self.expected_coords = self.stm.coords
        self.expected_values = [1., -1.]

    def test(self):
        result = temporal_correlation(self.stm, self.ref)
        assert_array_almost_equal(result.coords.x, self.expected_coords.x)
        assert_array_almost_equal(result.coords.y, self.expected_coords.y)
        assert_array_almost_equal(result.values, self.expected_values)


class TestTemporalCorrelationMatrix(unittest.TestCase):
    def setUp(self):
        self.stm = ScalarSpatioTemporalMap(coords=Coords2D(x=np.random.rand(3),
                                                           y=np.random.rand(3)),
                                           times=np.random.random(3),
                                           values=np.array([[1, 2, 3],
                                                            [4, 5, 6],
                                                            [9, 8, 7]]))

        self.expected = [[1., 1., -1.],
                         [1., 1., -1.],
                         [-1., -1., 1.]]

    def test(self):
        result = temporal_correlation_matrix(self.stm)
        assert_array_almost_equal(result, self.expected)

