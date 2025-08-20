from typing import Type, TypeVar, Union, cast, Dict, List
import inspect

import h5py
import numpy as np
from numpy.typing import NDArray



SupportedBaseTypes = Union[int, NDArray[np.float64], NDArray[np.object_]]

def _load_supported_base_type(hdf5_obj: h5py.Dataset) -> SupportedBaseTypes:
    if '__class__' not in hdf5_obj.attrs:
        raise ValueError("HDF5 object does not contain '__class__' attribute.")
    
    class_name = hdf5_obj.attrs['__class__'] #type: ignore
   
    if class_name == 'int':
        return int(hdf5_obj[()]) #type: ignore
    elif class_name == 'numpy.ndarray':
        return np.array(hdf5_obj[:]) #type: ignore
    else:
        raise ValueError(f"Unsupported class type '{class_name}' found in HDF5 object.")

def _save_supported_base_type(hdf5_obj: h5py.Group, name: str, value: SupportedBaseTypes) -> None:
    if isinstance(value, int):
        ds = hdf5_obj.create_dataset(name, data=value) #type: ignore
        ds.attrs.create('__class__', 'int') #type: ignore
    elif isinstance(value, np.ndarray):
        if value.dtype == np.float64:
            ds = hdf5_obj.create_dataset(name, data=value) #type: ignore
        elif value.dtype == np.object_:
            ds = hdf5_obj.create_dataset(name, data=value, dtype=h5py.vlen_dtype(np.float64)) #type: ignore
        else:
            raise ValueError(f"Unsupported numpy array dtype '{value.dtype}' for saving to HDF5.")
        ds.attrs.create('__class__', 'numpy.ndarray') #type: ignore
    else:
        raise ValueError(f"Unsupported variable '{name}' of type '{type(value)}' for saving to HDF5.")

T = TypeVar('T', bound='HDF5Mixin')

class HDF5Mixin:
    init_args: List[str] = []

    def __init_subclass__(cls):
        parameters = list(inspect.signature(cls.__init__).parameters.keys())
        cls.init_args = [p for p in parameters if not p.startswith('*') and p != 'self']

    @classmethod
    def from_hdf5(cls: Type[T], hdf5_obj: Union[h5py.File, h5py.Group, h5py.Dataset, h5py.Datatype]) -> T:
        if not isinstance(hdf5_obj, (h5py.Group, h5py.File)):
            raise TypeError("Expected hdf5_obj to be a Group or File.")
        kwargs: Dict[str, SupportedBaseTypes] = {}
        for k in cls.init_args:
            if k in hdf5_obj:
                v = hdf5_obj[k]
                kwargs[k] = _load_supported_base_type(cast(h5py.Dataset, v))
            else:
                raise KeyError(f"Missing required key '{k}' in HDF5 object.")
        return cls(**kwargs)

    def to_hdf5(self, hdf5_obj: Union[h5py.File, h5py.Group], name: str) -> None:
        group = hdf5_obj.create_group(name) # type: ignore
        for k in self.init_args:
            k_mod = k
            if not hasattr(self, k_mod):
                k_mod = f'_{k_mod}'
            if not hasattr(self, k_mod):
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{k}'")
            v = getattr(self, k_mod)
            _save_supported_base_type(group, k, v)

