import unittest

import numpy as np

from anaspike.dataclasses.coords2d import Coords2D



class TestCoords2D(unittest.TestCase):
    def test_initialization(self):
        coords = Coords2D([1, 2, 3], [4, 5, 6])
        self.assertEqual(len(coords), 3)
        np.testing.assert_array_almost_equal(coords.x, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(coords.y, [4.0, 5.0, 6.0])

    def test_invalid_initialization(self):
        with self.assertRaises(ValueError):
            Coords2D([1, 2], [3])  # Different lengths
        with self.assertRaises(ValueError):
            Coords2D([[1, 2]], [3, 4])  # x is not 1D
        with self.assertRaises(ValueError):
            Coords2D([1, 2], [[3, 4]])  # y is not 1D

    def test_properties(self):
        coords = Coords2D([1, 2], [3, 4])
        self.assertTrue(hasattr(coords, 'x'))
        self.assertTrue(hasattr(coords, 'y'))
        np.testing.assert_array_almost_equal(coords.x, [1.0, 2.0])
        np.testing.assert_array_almost_equal(coords.y, [3.0, 4.0])
        
    def test_hdf5(self):
        import h5py
        coords = Coords2D([1, 2, 3], [4, 5, 6])
        with h5py.File('test_coords2d.h5', 'w') as f_out:
            coords.to_hdf5(f_out, 'coords')
        with h5py.File('test_coords2d.h5', 'r') as f_in:
            coords_loaded = Coords2D.from_hdf5(f_in['coords'])

        self.assertEqual(len(coords_loaded), 3)
        np.testing.assert_array_almost_equal(coords_loaded.x, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(coords_loaded.y, [4.0, 5.0, 6.0])

