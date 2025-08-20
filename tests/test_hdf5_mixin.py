import unittest

import h5py

from anaspike.hdf5_mixin import HDF5Mixin



class TestHDF5Mixin(unittest.TestCase):
    def test_from_hdf5_missing_key(self):
        class DummyClass(HDF5Mixin):
            def __init__(self, value: int):
                self.value = value

        class DummyClass2(HDF5Mixin):
            def __init__(self, value: int):
                self.value2 = value

        instance = DummyClass(value=42)
        with h5py.File('test.h5', 'w') as f:
            instance.to_hdf5(f, 'dummy')

        with self.assertRaises(KeyError):
            DummyClass2.from_hdf5(f['dummy'])

    def test_to_hdf5_missing_member(self):
        class DummyClass(HDF5Mixin):
            def __init__(self, non_matching_arg: int):
                self.value = non_matching_arg

        instance = DummyClass(non_matching_arg=42)

        with h5py.File('test.h5', 'w') as f:
            with self.assertRaises(AttributeError):
                instance.to_hdf5(f, 'dummy')

    def test_int_member(self):
        class DummyClass(HDF5Mixin):
            def __init__(self, value: int):
                self.value = value
                

        instance = DummyClass(value=42)
        with h5py.File('test.h5', 'w') as f:
            instance.to_hdf5(f, 'dummy')

        with h5py.File('test.h5', 'r') as f:
            loaded_instance = DummyClass.from_hdf5(f['dummy'])

        self.assertEqual(instance.value, loaded_instance.value)

    def test_ndarray_member(self):
        import numpy as np
        from numpy.typing import NDArray
        class DummyClass(HDF5Mixin):
            def __init__(self, array: NDArray[np.float64]):
                self.array = array

        array_data = np.array([1.0, 2.0, 3.0])
        instance = DummyClass(array=array_data)

        with h5py.File('test.h5', 'w') as f:
            instance.to_hdf5(f, 'dummy')

        with h5py.File('test.h5', 'r') as f:
            loaded_instance = DummyClass.from_hdf5(f['dummy'])

        np.testing.assert_array_equal(instance.array, loaded_instance.array)
