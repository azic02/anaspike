from typing import Type, TypeVar, Union, cast, Dict, List, Tuple
import inspect

import h5py
import numpy as np
from numpy.typing import NDArray



SupportedBaseTypes = Union[float, int, NDArray[np.int64], NDArray[np.float64], NDArray[np.object_],
                           'HDF5Mixin']
SupportedContainerTypes = Union[List[SupportedBaseTypes], Tuple[SupportedBaseTypes, ...]]

AllSupportedTypes = Union[SupportedBaseTypes, SupportedContainerTypes]


_REGISTRY: Dict[str, Type['HDF5Mixin']] = {}


def _load_supported_base_type(hdf5_obj: h5py.Dataset) -> SupportedBaseTypes:
    if '__class__' not in hdf5_obj.attrs:
        raise ValueError("HDF5 object does not contain '__class__' attribute.")
    
    class_name = hdf5_obj.attrs['__class__'] #type: ignore
   
    if class_name == 'int':
        return int(hdf5_obj[()]) #type: ignore
    elif class_name == 'float':
        return float(hdf5_obj[()]) #type: ignore
    elif class_name == 'numpy.ndarray':
        return np.array(hdf5_obj[:]) #type: ignore
    elif class_name in _REGISTRY:
        cls = _REGISTRY[class_name]
        return cls.from_hdf5(cast(h5py.Group, hdf5_obj))
    else:
        raise ValueError(f"Unsupported class type '{class_name}' found in HDF5 object.")

def _load_supported_container_type(hdf5_obj: h5py.Group) -> SupportedContainerTypes:
    if '__class__' not in hdf5_obj.attrs:
        raise ValueError("HDF5 object does not contain '__class__' attribute.")
    
    idxs: List[int] = sorted(hdf5_obj.keys(), key=lambda x: int(x)) #type: ignore
    class_name = hdf5_obj.attrs['__class__'] #type: ignore
    if class_name == 'list':
        return [_load_supported_base_type(cast(h5py.Dataset, hdf5_obj[idx])) for idx in idxs]
    elif class_name == 'tuple':
        return tuple(_load_supported_base_type(cast(h5py.Dataset, hdf5_obj[idx])) for idx in idxs)
    else:
        raise ValueError(f"Unsupported container type '{class_name}' found in HDF5 object.")

def _load_supported_type(hdf5_obj: Union[h5py.Group, h5py.Dataset]) -> AllSupportedTypes:
    if '__metaclass__' not in hdf5_obj.attrs:
        raise ValueError("HDF5 object does not contain '__metaclass__' attribute.")

    metaclass = hdf5_obj.attrs['__metaclass__'] #type: ignore
    if metaclass == 'base':
        return _load_supported_base_type(cast(h5py.Dataset, hdf5_obj))
    elif metaclass == 'container':
        return _load_supported_container_type(cast(h5py.Group, hdf5_obj))
    else:
        raise ValueError(f"Unsupported metaclass '{metaclass}' found in HDF5 object.")

def _save_supported_type(hdf5_obj: h5py.Group, name: str, value: AllSupportedTypes) -> None:
    if isinstance(value, int):
        ds = hdf5_obj.create_dataset(name, data=value) #type: ignore
        ds.attrs.create('__metaclass__', 'base') #type: ignore
        ds.attrs.create('__class__', 'int') #type: ignore
    elif isinstance(value, float):
        ds = hdf5_obj.create_dataset(name, data=value) #type: ignore
        ds.attrs.create('__metaclass__', 'base') #type: ignore
        ds.attrs.create('__class__', 'float') #type: ignore
    elif isinstance(value, np.ndarray):
        if value.dtype == np.float64:
            ds = hdf5_obj.create_dataset(name, data=value) #type: ignore
        elif value.dtype == np.int64:
            ds = hdf5_obj.create_dataset(name, data=value) #type: ignore
        elif value.dtype == np.object_:
            ds = hdf5_obj.create_dataset(name, data=value, dtype=h5py.vlen_dtype(np.float64)) #type: ignore
        else:
            raise ValueError(f"Unsupported numpy array dtype '{value.dtype}' for saving to HDF5.")
        ds.attrs.create('__metaclass__', 'base') #type: ignore
        ds.attrs.create('__class__', 'numpy.ndarray') #type: ignore
    elif isinstance(value, HDF5Mixin):
        group = value.to_hdf5(hdf5_obj, name)
        group.attrs.create('__metaclass__', 'base') #type: ignore
        group.attrs.create('__class__', value.__class__.__name__) #type: ignore
    elif isinstance(value, list):
        sub_group = hdf5_obj.create_group(name) #type: ignore
        sub_group.attrs.create('__metaclass__', 'container') #type: ignore
        sub_group.attrs.create('__class__', 'list') #type: ignore
        for i, item in enumerate(value):
            _save_supported_type(sub_group, str(i), item)
    elif isinstance(value, tuple):
        sub_group = hdf5_obj.create_group(name) #type: ignore
        sub_group.attrs.create('__metaclass__', 'container') #type: ignore
        sub_group.attrs.create('__class__', 'tuple') #type: ignore
        for i, item in enumerate(value):
            _save_supported_type(sub_group, str(i), item)
    else:
        raise ValueError(f"Unsupported variable '{name}' of type '{type(value)}' for saving to HDF5.")

T = TypeVar('T', bound='HDF5Mixin')

class HDF5Mixin:
    init_args: List[str] = []

    def __init_subclass__(cls):
        parameters = list(inspect.signature(cls.__init__).parameters.keys())
        cls.init_args = [p for p in parameters if not p.startswith('*') and p != 'self']
        _REGISTRY[cls.__name__] = cls

    @classmethod
    def from_hdf5(cls: Type[T], hdf5_obj: Union[h5py.File, h5py.Group, h5py.Dataset, h5py.Datatype]) -> T:
        if not isinstance(hdf5_obj, (h5py.Group, h5py.File)):
            raise TypeError("Expected hdf5_obj to be a Group or File.")
        kwargs: Dict[str, AllSupportedTypes] = {}
        for k in cls.init_args:
            if k in hdf5_obj:
                v = hdf5_obj[k]
                kwargs[k] = _load_supported_type(cast(h5py.Dataset, v))
            else:
                raise KeyError(f"Missing required key '{k}' in HDF5 object.")
        return cls(**kwargs)

    def to_hdf5(self, hdf5_obj: Union[h5py.File, h5py.Group], name: str) -> h5py.Group:
        group = hdf5_obj.create_group(name) # type: ignore
        for k in self.init_args:
            k_mod = k
            if not hasattr(self, k_mod):
                k_mod = f'_{k_mod}'
            if not hasattr(self, k_mod):
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{k}'")
            v = getattr(self, k_mod)
            _save_supported_type(group, k, v)
        return group

