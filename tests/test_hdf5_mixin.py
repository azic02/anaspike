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

    def test_ndarray_dtype_float_member(self):
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
        self.assertIs(loaded_instance.array.dtype, np.dtype(np.float64))

    def test_ndarray_dtype_int_member(self):
        import numpy as np
        from numpy.typing import NDArray

        class DummyClass(HDF5Mixin):
            def __init__(self, array: NDArray[np.int64]):
                self.array = array

        array_data = np.array([1, 2, 3], dtype=np.int64)
        instance = DummyClass(array=array_data)

        with h5py.File('test.h5', 'w') as f:
            instance.to_hdf5(f, 'dummy')

        with h5py.File('test.h5', 'r') as f:
            loaded_instance = DummyClass.from_hdf5(f['dummy'])

        np.testing.assert_array_equal(instance.array, loaded_instance.array)
        self.assertIs(loaded_instance.array.dtype, np.dtype(np.int64))

    def test_underscored_member(self):
        class DummyClass(HDF5Mixin):
            def __init__(self, value: int):
                self._value = value

            def get_value(self):
                return self._value

        instance = DummyClass(value=42)
        with h5py.File('test.h5', 'w') as f:
            instance.to_hdf5(f, 'dummy')

        with h5py.File('test.h5', 'r') as f:
            loaded_instance = DummyClass.from_hdf5(f['dummy'])

        self.assertEqual(instance.get_value(), loaded_instance.get_value())

    def test_ndarray_of_ndarray_of_differing_lengths_member(self):
        import numpy as np
        from numpy.typing import NDArray

        class DummyClass(HDF5Mixin):
            def __init__(self, arrays: list[NDArray[np.float64]]):
                self.arrays = np.array(arrays, dtype=object)

        arrays_data = [np.array([1.0, 2.0]), np.array([3.0])]
        instance = DummyClass(arrays=arrays_data)

        with h5py.File('test.h5', 'w') as f:
            instance.to_hdf5(f, 'dummy')

        with h5py.File('test.h5', 'r') as f:
            loaded_instance = DummyClass.from_hdf5(f['dummy'])

        for original, loaded in zip(instance.arrays, loaded_instance.arrays):
            np.testing.assert_array_equal(original, loaded)

    def test_to_hdf5_ndarray_unsupported_dtype(self):
        import numpy as np
        from numpy.typing import NDArray

        class DummyClass(HDF5Mixin):
            def __init__(self, array: NDArray[np.str_]):
                self.array = array

        instance = DummyClass(array=np.array(['a', 'b', 'c'], dtype=np.str_))
        with h5py.File('test.h5', 'w') as f:
            with self.assertRaises(ValueError):
                instance.to_hdf5(f, 'dummy')

    def test_two_simultaneous_classes(self):
        class ClassA(HDF5Mixin):
            def __init__(self, valueA: int):
                self.valueA = valueA
        class ClassB(HDF5Mixin):
            def __init__(self, valueB: int):
                self.valueB = valueB

        instanceA = ClassA(valueA=10)
        instanceB = ClassB(valueB=20)
        with h5py.File('test.h5', 'w') as f:
            instanceA.to_hdf5(f, 'classA')
            instanceB.to_hdf5(f, 'classB')
        with h5py.File('test.h5', 'r') as f:
            loaded_instanceA = ClassA.from_hdf5(f['classA'])
            loaded_instanceB = ClassB.from_hdf5(f['classB'])

        self.assertEqual(instanceA.valueA, loaded_instanceA.valueA)
        self.assertEqual(instanceB.valueB, loaded_instanceB.valueB)

    def test_nested_hdf5mixin_member(self):
        class InnerClass(HDF5Mixin):
            def __init__(self, value: int):
                self.value = value

        class OuterClass(HDF5Mixin):
            def __init__(self, inner: InnerClass):
                self.inner = inner

        inner_instance = InnerClass(value=42)
        outer_instance = OuterClass(inner=inner_instance)

        with h5py.File('test.h5', 'w') as f:
            outer_instance.to_hdf5(f, 'outer')

        with h5py.File('test.h5', 'r') as f:
            loaded_outer = OuterClass.from_hdf5(f['outer'])

        self.assertEqual(outer_instance.inner.value, loaded_outer.inner.value)

