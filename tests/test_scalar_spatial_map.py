import unittest

import numpy as np

from anaspike.dataclasses.scalar_spatial_map import ScalarSpatialMap



class TestInit(unittest.TestCase):
    def setUp(self):
        from anaspike.dataclasses.coords2d import Coords2D
        self.coords = Coords2D(x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 2.0])
    def test_valid_init(self):
        fr = ScalarSpatialMap(self.coords, values=np.array([0.0, 1.0, 2.0]))
        self.assertIsInstance(fr, ScalarSpatialMap)
        np.testing.assert_array_almost_equal(fr.values, np.array([0.0, 1.0, 2.0]))

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            ScalarSpatialMap(
                    coords=self.coords,
                    values=np.array([[0.0, 1.0],
                                     [2.0, 3.0]]))


