# Important constants about the project are here, mainly paths of important
# files/folders.

import os
import shutil
import pickle as pkl
import csv
import math
import textwrap
from functools import cached_property
from typing import List, Dict, Tuple, Iterable
from typing import Optional, Union, Any, Literal

import numpy as np
import pandas as pd

import yaml

from .common_utils import get_image
from . import image_utils as imu
# from .common_utils import bu.np_to_pil, bu.pil_to_np, bu.np_to_torch, bu.torch_to_np
from .common_types import *


# commonly used types
ImageFormat = Literal['pil', 'np', 'torch']
File = Union['CSVFile', 'PKLFile', 'NPYFile', 'PNGFile']


# io functions
def load_obj(name: str) -> Any:
    with open(name, 'rb') as f:
        return pkl.load(f)

def save_obj(name: str, obj: Any) -> None:
    with open(name, 'wb') as f:
        pkl.dump(obj, f)
        
def load_yaml(path: str) -> Any:
    
    with open(path) as f:
        obj = yaml.safe_load(f)
    
    return obj
        
def npy_shape(name: str) -> Tuple[int, ...]:
    '''Reads the shape of a numpy array saved as an .npy file.'''
    
    with open(name, 'rb') as f:
        magic = np.lib.format.read_magic(f)
        version = str(magic[0])
        sub_version = str(magic[1])

        header_reader = 'read_array_header_' + version + '_' + sub_version
        header_reader = getattr(np.lib.format, header_reader)

        shape = header_reader(f)[0]
    
    return shape

def listdir(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError()

    return list(map(
        lambda dir: os.path.join(path, dir),
        os.listdir(path)
    ))


class Base(os.PathLike):
    __ext_registry: Dict[str, type] = {}

    def __init_subclass__(cls, ext: str, *args, **kwargs) -> None:

        Base.__ext_registry[ext] = cls

    def __new__(cls, name: str, *args, **kwargs) -> 'Base':

        _, ext = os.path.splitext(name)
        subcls = Base.__ext_registry[ext]
        obj = object.__new__(subcls)
        return obj

    def __init__(self, name: str, parent : Optional['Folder'] = None) -> None:

        self.name = name
        self.parent = parent

        if self.parent is not None:
            self.parent.children[name] = self

    def __fspath__(self) -> str:
        return self.path

    def __str__(self) -> str:
        return self.path

    @cached_property
    def path(self) -> str:
        if self.parent is None:
            return self.name

        return os.path.join(self.parent.path, self.name)

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def content(self, depth : int = math.inf) -> str:
        string = self.name + '\n'

        if depth == 0:
            return string

        if hasattr(self, 'children'):
            for child in self.children.values():
                temp = child.content(depth=depth-1)
                temp = textwrap.indent(temp, Base.indent)
                string += temp

        return string

    def delete(self) -> None:
        if not self.exists():
            raise FileNotFoundError('{} does not exist.'.format(self.path))

        try:
            shutil.rmtree(self.path)
        except NotADirectoryError:
            os.remove(self.path)

class Folder(Base, ext=''):
    def __init__(self, name: str, parent: Optional['Folder'] = None) -> None:
        
        super().__init__(name, parent=parent)

        self.children = {}
        
    def __getitem__(self, name: str) -> Union['Folder', File]:
        
        name = str(name)
        
        if name in self.children:
            return self.children[name]

        # if there is no child with the given name, create one
        new_child = self.add(name)
        return new_child
        
    def __iter__(self) -> Iterable['Base']:
        
        return iter(self.children.values())
        
    def __len__(self) -> int:
        
        return len(self.children)

    def add(self, child: Union[str, 'Base']) -> 'Base':

        if isinstance(child, str):
            child = Base(child)

        self.children[child.name] = child

        if child.parent is not None:
            del child.parent.children[child.name]

        child.parent = self

        return child

    def discard(self, child: str) -> None:
        if child.parent != self:
            raise Exception('NoChildError')

        child.parent = None
        del self.children[child.name]

    def create(self) -> None:
        if self.exists():
            raise FileExistsError('{} is already exists.'.format(self.path))

        os.makedirs(self.path)

    def delete(self) -> None:
        if not self.exists():
            raise FileNotFoundError('{} does not exist.'.format(self.path))

        for child in self.children:
            if child.exists():
                child.delete()

        os.rmdir(self.path)

    def physical_children(self) -> List[str]:
        
        if not self.exists():
            raise FileNotFoundError
        
        return os.listdir(self.path)


class CSVFile(Base, ext='.csv'):
    def __init__(
        self, name: str, parent: Optional['Folder'] = None, 
        fieldnames: Optional[List[str]] = None
    ) -> None:
        
        super().__init__(name, parent=parent)

        self.fieldnames = fieldnames

    def __len__(self) -> int:
        
        return len(list(self))

    def __iter__(self) -> Iterable[str]:
        
        with open(self.path, 'r', newline='') as f:
            if self.fieldnames is None:
                reader = csv.reader(f)
            else:
                reader = csv.DictReader(f, self.fieldnames)

            for row in reader:
                yield row

    def save(self, dataframe: DataFrame, append=False) -> None:
        
        if self.parent is not None and not self.parent.exists():
            self.parent.create()

        mode = 'a' if append else 'w'

        header = False
        if not append:
            header = True
        elif not self.exists():
            header = True
        elif os.path.getsize(self.path) <= 0:
            header = True

        dataframe.to_csv(self.path, mode=mode, header=header, index=True)

    def load(self):
        
        df = pd.read_csv(self.path)

        df = df.set_index('model name')
        
        df = df.convert_dtypes()
        # df = df.astype('float', errors='ignore')

        return df

class TXTFile(Base, ext='.txt'):
    def __init__(
        self, name: str, parent: Optional['Folder'] = None
    ) -> None:
        
        super().__init__(name, parent=parent)
    
    def save(self, txt: str, append: bool = False) -> None:
        
        if self.parent is not None and not self.parent.exists():
            self.parent.create()

        mode = 'a' if append else 'w'

        txt = str(txt)
        with open(self.path, mode=mode) as f:
            f.write(txt)

    def load(self):
        
        with open(self.path) as f:
            s = f.read()
        
        return s

class LSTFile(Base, ext='.lst'):
    def __init__(
        self, name: str, parent: Optional['Folder'] = None
    ) -> None:
        
        super().__init__(name, parent=parent)
        
    def _to_list(self) -> List[str]:
        
        with open(self.path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        return lines

    def __len__(self) -> int:
        
        return len(self._to_list())

    def __iter__(self) -> Iterable[str]:
        
        return iter(self._to_list())
        

    def save(self, lines: List[str], append=False) -> None:
        
        if self.parent is not None and not self.parent.exists():
            self.parent.create()

        mode = 'a' if append else 'w'

        with open(self.path, mode=mode) as f:
            f.write('\n'.join(lines))

    def load(self):
        
        return list(self)


class YAMLFile(Base, ext='.yaml'):
    def __init__(
        self, name: str, parent: Optional['Folder'] = None
    ) -> None:
        
        super().__init__(name, parent=parent)

    def load(self) -> Dict[str, Any]:
        
        return load_yaml(self.path)

    def save(self, obj: Any, overwrite: bool = True) -> None:
        
        NotImplemented

class PKLFile(Base, ext='.pkl'):
    def __init__(
        self, name: str, parent: Optional['Folder'] = None
    ) -> None:
        
        super().__init__(name, parent=parent)

    def load(self) -> Any:
        
        return load_obj(self.path)

    def save(self, obj: Any, overwrite: bool = True) -> None:
        
        if not overwrite and self.exists():
            raise FileExistsError

        if self.parent is not None and not self.parent.exists():
            self.parent.create()

        save_obj(self.path, obj)

class NPYFile(Base, ext='.npy'):
    def __init__(
        self, name: str, parent: Optional['Folder'] = None
    ) -> None:
        
        super().__init__(name, parent=parent)

    def load(self, mmap_mode=None) -> NumpyArray:
        
        return np.load(self.path, mmap_mode=mmap_mode)

    def save(self, arr: NumpyArray, overwrite: bool = True) -> None:
        
        if not overwrite and self.exists():
            raise FileExistsError

        if self.parent is not None and not self.parent.exists():
            self.parent.create()

        np.save(self.path, arr, )

    @property
    def shape(self) -> Tuple[int, ...]:
        
        return npy_shape(self.path)

class PNGFile(Base, ext='.png'):
    def __init__(
        self, name: str, parent: Optional['Folder'] = None
    ) -> None:
        
        super().__init__(name, parent=parent)

    def load(
        self, format: ImageFormat = 'np', 
        size: Union[Tuple[int, int], int] = -1, 
        d: int = 32
    ) -> Union[PILImg, NumpyArray, Tensor]:
        
        if format == 'pil':
            img_pil = get_image(self.path, imsize=size)[0]
            # img_pil = crop_image(img_pil, d=d)
            return img_pil

        if format == 'np':
            img_pil = get_image(self.path, imsize=size)[0]
            # img_pil = crop_image(img_pil, d=d)
            img_np = imu.pil_to_np(img_pil)
            return img_np

        if format == 'torch':
            img_pil = get_image(self.path, imsize=size)[0]
            # img_pil = crop_image(img_pil, d=d)
            img_np = imu.pil_to_np(img_pil)
            img_torch = imu.np_to_torch(img_np)
            return img_torch

    def save(
        self, img: Img, 
        format: ImageFormat = 'np', overwrite: bool = True
    ) -> None:
        
        if not overwrite and self.exists():
            raise FileExistsError

        if self.parent is not None and not self.parent.exists():
            self.parent.create()

        if format == 'pil':
            img.save(self.path)

        if format == 'np':
            img = imu.np_to_pil(img)
            img.save(self.path)

        if format == 'torch':
            img = imu.torch_to_np(img)
            img = imu.np_to_pil(img)
            img.save(self.path)



# important project constants
NUM_RANDOM_OUTPUTS = 100
NUM_HIGH_LR_OUTPUTS = 10




# change this to the path of the current directory
PROJECT_FOLDER = \
    '/home/ersin/Documents/machine learning/NAS-DIP Summer Research'




IMG_SIZE = 512
IMG_EXT = '.png'

# commonly used folders and files
ROOT = Folder(PROJECT_FOLDER)

benchmark = ROOT['benchmark']

images = ROOT['images']

denoising_images = images['denoising']
inpainting_images = images['inpainting']

denoising_data = benchmark['denoising']
denoising_data = benchmark['denoising']



models_csv = ROOT['benchmark']['models.csv']
model_names = [f'0_{t[0]}_iteration_4000_sigma_25_skip_{t[1]}' for t in models_csv]
model_names_hashed = {name: False for name in model_names}
num_models = len(model_names)



