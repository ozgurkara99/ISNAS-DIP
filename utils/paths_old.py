# Important constants about the project are here, mainly paths of important
# files/folders.

import os
import csv
from functools import cached_property
from collections import namedtuple
from math import inf
import textwrap

import numpy as np
import pandas as pd

from .common_utils import get_image, crop_image
from .common_utils import np_to_pil, pil_to_np, np_to_torch, torch_to_np
from .basic_utils import load_obj, save_obj, npy_shape



# important project constants
NUM_RANDOM_OUTPUTS = 100
PROJECT_FOLDER = \
    '/srv/beegfs02/scratch/biwismrschool21/data/NAS-DIP Summer Research'

IMG_SIZE = 512
IMG_EXT = '.png'

# to be used in metric calculations
HIST_BINS = 100
HIST_DENSITY = True
HIST_RANGE = None




indent = ' '*3


class Base(os.PathLike):
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent

        if self.parent is not None:
            self.parent.children[name] = self

    @cached_property
    def path(self):
        if self.parent is None:
            return self.name

        return os.path.join(self.parent.path, self.name)

    def __str__(self):
        return self.path

    def __fspath__(self):
        return self.path

    def exists(self):
        return os.path.exists(self.path)

    def content(self, depth=inf):
        string = self.name + '\n'

        if depth == 0:
            return string
        
        if hasattr(self, 'children'):
            for child in self.children.values():
                temp = child.content(depth=depth-1)
                temp = textwrap.indent(temp, indent)
                string += temp
        
        return string


class Folder(Base):
    def __init__(self, name, parent=None, children=[]):
        super().__init__(name, parent=parent)

        self.children = {}

        self.add(*children)
        
    def __getattr__(self, name):
        if name in self.children:
            return self.children[name]

        for child_name, child in self.children.items():
            child_name = list(child_name)

            for i, c in enumerate(child_name):
                if c in (' ', '.', ',', '-', '+', '?', '!', ':', ';'):
                    child_name[i] = '_'
            
            child_name = ''.join(child_name)

            if child_name == name:
                return child
        
        raise AttributeError

    def __getitem__(self, name):
        return self.children[name]

    def add(self, *children):
        for child in children:
            if isinstance(child, str):
                base, ext = os.path.splitext(child)

                if len(ext) == 0:
                    child = Folder(child)
                else:
                    if ext == '.csv':
                        child = CSVFile(child)
                    elif ext == '.pkl':
                        child = PKLFile(child)
                    elif ext == '.npy':
                        child = NPYFile(child)
                    elif ext == '.png':
                        child = PNGFile(child)
                    else:
                        child = File(child)

            self.children[child.name] = child

            if child.parent is not None:
                del child.parent.children[child.name]
            
            child.parent = self
    
    def discard(self, *children):
        for child in children:
            if child.parent != self:
                raise Exception('NoChildError')
            
            child.parent = None
            del self.children[child.name]
    
    def create(self):
        if self.exists():
            raise FileExistsError('{} is already exists.'.format(self.path))

        os.makedirs(self.path)

    def delete(self):
        if not self.exists():
            raise FileNotFoundError('{} does not exist.'.format(self.path))

        for child in self.children:
            if child.exists():
                child.delete()
        
        os.rmdir(self.path)
    
    def __iter__(self):
        return iter(self.children.values())

    def __len__(self):
        return len(self.children)


class File(Base):
    def __init__(self, name, parent=None):
        super().__init__(name, parent=parent)

    def delete(self):
        if not self.exists():
            raise FileNotFoundError('{} does not exist.'.format(self.path))

        os.remove(self.path)

    def ext(self):
        return os.path.splitext(self.name)[1]
    
    def stem(self):
        return os.path.splitext(self.name)[0]

    def save(self):
        if self.parent is not None and not self.parent.exists():
            self.parent.create()


class CSVFile(File):
    def __init__(self, name, parent=None, fieldnames=None):
        super().__init__(name, parent=parent)

        self.fieldnames = fieldnames
    
    def __iter__(self):
        with open(self.path, 'r', newline='') as f:
            if self.fieldnames is None:
                reader = csv.reader(f)
            else:
                reader = csv.DictReader(f, self.fieldnames)
            
            for row in reader:
                yield row
    
    def save(self, dataframe, append=False):
        super().save()

        # save_csv(self.path, rows, self.fieldnames, append=append)

        mode = 'a' if append else 'w'

        header = False
        if not append:
            header = True
        elif not self.exists():
            header = True
        elif os.path.getsize(self.path) <= 0:
            header = True

        dataframe.to_csv(self.path, mode=mode, header=header, index=False)

    def load(self):
        # return load_csv(self.path)
        return pd.read_csv(self.path)

    def __len__(self):
        return len(list(self))


class PKLFile(File):
    def __init__(self, name, parent=None):
        super().__init__(name, parent=parent)

    def load(self):
        return load_obj(self.path)

    def save(self, obj, overwrite=True):
        if not overwrite and self.exists():
            raise FileExistsError

        super().save()

        save_obj(self.path, obj)


class NPYFile(File):
    def __init__(self, name, parent=None):
        super().__init__(name, parent=parent)

    def load(self, mmap_mode=None):
        return np.load(self.path, mmap_mode=mmap_mode)

    def save(self, arr, overwrite=True):
        if not overwrite and self.exists():
            raise FileExistsError

        super().save()

        np.save(self.path, arr, )

    @property
    def shape(self):
        return npy_shape(self.path)


class PNGFile(File):
    def __init__(self, name, parent=None):
        super().__init__(name, parent=parent)

    def load(self, format='np', size=-1, d=32):
        if not format in ('pil', 'np', 'torch'):
            raise Exception('Invalid format, given {}'.format(format))


        if format == 'pil':
            img_pil = get_image(self.path, imsize=size)[0]
            img_pil = crop_image(img_pil, d=d)
            return img_pil

        if format == 'np':
            img_pil = get_image(self.path, imsize=size)[0]
            img_pil = crop_image(img_pil, d=d)
            img_np = pil_to_np(img_pil)
            return img_np

        if format == 'torch':
            img_pil = get_image(self.path, imsize=size)[0]
            img_pil = crop_image(img_pil, d=d)
            img_np = pil_to_np(img_pil)
            img_torch = np_to_torch(img_np)
            return img_torch

    def save(self, img, format='np', overwrite=True):
        if not overwrite and self.exists():
            raise FileExistsError

        if not format in ('pil', 'np', 'torch'):
            raise Exception('Invalid format, given {}'.format(format))

        super().save()

        if format == 'pil':
            img.save(self.path)
        
        if format == 'np':
            img = np_to_pil(img)
            img.save(self.path)

        if format == 'torch':
            img = torch_to_np(img)
            img = np_to_pil(img)
            img.save(self.path)



# We represent the project folder with a Folder object
root = Folder(PROJECT_FOLDER)

root.add('benchmark')
root.add('images')

root.benchmark.add('denoising')
root.benchmark.add('inpainting')

# maybe, you can add field names for this CSVFile object
root.benchmark.add('lowpass_metrics.csv')

root.benchmark.add('models.csv')

if not root.benchmark.models_csv.exists():
    raise FileNotFoundError(
        'There is no {} file.'.format(root.benchmark.models_csv.path())
    )

model_names = list(map(
    lambda t: '0_{}_iteration_4000_sigma_25_skip_{}'.format(*t),
    root.benchmark.models_csv
))
model_names_hashed = {name: False for name in model_names}
num_models = len(model_names)

root.benchmark.add('random_outputs')

for model_name in model_names:
    root.benchmark.random_outputs.add(model_name)

    root.benchmark.random_outputs[model_name].add('random_output.npy')

    # may be delete this for loop in the future
    for i in range(NUM_RANDOM_OUTPUTS):
        root.benchmark.random_outputs[model_name].add(
            'random_output_{:04}.npy'.format(i)
        )


# create the denoising images folder
root.images.add('denoising')

if not root.images.denoising.exists():
    raise FileNotFoundError(
        'There is no {} folder.'.format(root.images.denoising.path())
    )

for fname in os.listdir(root.images.denoising.path):
    stem, ext = os.path.splitext(fname)

    if ext != '.png':
        continue

    # fill the denoising folder inside benchmark
    root.images.denoising.add(fname)

    root.benchmark.denoising.add(stem)

    root.benchmark.denoising[stem].add('data')

    for model_name in model_names:
        root.benchmark.denoising[stem].data.add(model_name)
        root.benchmark.denoising[stem].data[model_name].add('res.pkl')
        root.benchmark.denoising[stem].data[model_name].add('grid.png')

    # maybe, you can add field names for this CSVFile object
    root.benchmark.denoising[stem].add('psnr.csv')
    root.benchmark.denoising[stem].add('similarity_metrics.csv')


# create the inpainting images folder
root.images.add('inpainting')

if not root.images.inpainting.exists():
    raise FileNotFoundError(
        'There is no {} folder.'.format(root.images.inpainting.path())
    )

for fname in os.listdir(root.images.inpainting.path):
    stem, ext = os.path.splitext(fname)

    if ext != '.png':
        continue

    # fill the inpainting folder inside benchmark
    root.images.inpainting.add(fname)

    root.benchmark.inpainting.add(stem)

    root.benchmark.inpainting[stem].add('data')

    for model_name in model_names:
        root.benchmark.inpainting[stem].data.add(model_name)
        root.benchmark.inpainting[stem].data[model_name].add('res.pkl')
        root.benchmark.inpainting[stem].data[model_name].add('grid.png')

    # maybe, you can add field names for this CSVFile object
    root.benchmark.inpainting[stem].add('psnr.csv')
    root.benchmark.inpainting[stem].add('similarity_metrics.csv')


if __name__ == '__main__':
    
    def print_exist(folder):
        print('{} exists.'.format(folder.path))
    
    def print_not_exists(folder):
        print('{} does NOT exists.'.format(folder.path))

    def print_existance(folder):
        if folder.exists():
            print_exist(folder)
        else:
            print_not_exists(folder)

    

    print('Checking the integrity of the project directory...\n')


    if not root.exists():
        print_not_exists(root)
        exit()
    

    if not root.images.exists():
        print_not_exists(root.images)
        exit()
    

    if not root.images.denoising.exists():
        print_not_exists(root.images.denoising)
        exit()
    else:
        print('Denoising Images: ')
        for file in root.images.denoising:
            print(file.path)
        print()


    if not root.benchmark.exists():
        print_not_exists(root.benchmark)
        exit()


    if not root.benchmark.models_csv.exists():
        print_not_exists(root.benchmark.models_csv)
        exit()
    else:
        print('There are {} models inside {}\n'.format(
            num_models, root.benchmark.models_csv.path
        ))


    if not root.benchmark.lowpass_metrics_csv.exists():
        print_not_exists(root.benchmark.lowpass_metrics_csv)
        print()
    else:
        for row in root.benchmark.lowpass_metrics_csv:
            model_name = row[0]

            # we are assuming that the field of the first column will
            # always be model_name
            if model_name == 'model_name':
                continue

            if not model_name in model_names_hashed:
                print('{} exists in {} but it should NOT.'.format(
                    model_name, root.benchmark.lowpass_metrics_csv
                ))
                continue

            model_names_hashed[model_name] = True
        print()
        
        for model_name, val in model_names_hashed.items():
            if not val:
                print('{} does NOT exists in {} but it should.'.format(
                    model_name, root.benchmark.lowpass_metrics_csv
                ))
            
            model_names_hashed[model_name] = False
        print()
    

    if not root.benchmark.random_outputs.exists():
        print_not_exists(root.benchmark.random_outputs)
    else:
        for folder in os.listdir(root.benchmark.random_outputs.path):
            if not folder in model_names_hashed:
                print('{} exists in {} but it should NOT.'.format(
                    os.path.join(root.benchmark.random_outputs.path, folder),
                    root.benchmark.random_outputs.path
                ))
        print()
            
        for child in root.benchmark.random_outputs:
            if not child.exists():
                print('{} does NOT exists in {} but it should.'.format(
                    child.path, root.benchmark.random_outputs.path
                ))
                print()
                continue
            
            if not child.random_output_npy.exists():
                print('{} does NOT exists in {} but it should.'.format(
                    child.path, root.benchmark.random_outputs.path
                ))
                print()
                continue
            
            shape = child.random_output_npy.shape
            true_shape = (NUM_RANDOM_OUTPUTS, 1, IMG_SIZE, IMG_SIZE)
            if shape != true_shape:
                print('Shape of {} is {} which FALSE. It should be {}.'.format(
                    child.random_output_npy.path, shape, true_shape
                ))
                print()
        print()
    

    if not root.benchmark.denoising.exists():
        print_not_exists(root.benchmark.denoising)
        print()
    else:
        for img_folder in root.benchmark.denoising:
            if not img_folder.exists():
                print_not_exists(img_folder)
                print()
                continue
            
            if not img_folder.data.exists():
                print_not_exists(img_folder.data)
                print()
            else:
                for folder in os.listdir(img_folder.data.path):
                    if not folder in model_names_hashed:
                        print('{} exists in {} but it should NOT.'.format(
                            os.path.join(img_folder.data.path, folder), 
                            img_folder.data.path
                        ))
                print()
                    
                for child in img_folder.data:
                    if not child.exists():
                        print('{} does NOT exists in {} but it should.'.format(
                            child.path, img_folder.data.path
                        ))
                        print()
                        continue
                    
                    if not child.grid_png.exists():
                        print('{} does NOT exists in {} but it should.'.format(
                            child.grid_png.path, child.path
                        ))
                    
                    if not child.res_pkl.exists():
                        print('{} does NOT exists in {} but it should.'.format(
                            child.res_pkl.path, child.path
                        ))
                        print()
                        continue
                    
                    # check whether res.pkl contains necessary information
            

            if not img_folder.psnr_csv.exists():
                print_not_exists(img_folder.psnr_csv)
                print()
            else:
                for row in list(img_folder.psnr_csv):
                    model_name = row[0]

                    # we are assuming that the field of the first column will
                    # always be model_name
                    if model_name == 'model_name':
                        continue

                    if not model_name in model_names_hashed:
                        print('{} exists in {} but it should NOT.'.format(
                            model_name, img_folder.psnr_csv
                        ))
                        print()
                        continue

                    model_names_hashed[model_name] = True
                
                for model_name, val in model_names_hashed.items():
                    if not val:
                        print('{} does NOT exists in {} but it should.'.format(
                            model_name, img_folder.psnr_csv
                        ))

                    model_names_hashed[model_name] = False


            if not img_folder.similarity_metrics_csv.exists():
                print_not_exists(img_folder.similarity_metrics_csv)
                print()
            else:
                for row in img_folder.similarity_metrics_csv:
                    model_name = row[0]

                    # we are assuming that the field of the first column will
                    # always be model_name
                    if model_name == 'model_name':
                        continue

                    if not model_name in model_names_hashed:
                        print('{} exists in {} but it should NOT.'.format(
                            model_name, img_folder.similarity_metrics_csv
                        ))
                        print()
                        continue

                    model_names_hashed[model_name] = True
                
                for model_name, val in model_names_hashed.items():
                    if not val:
                        print('{} does NOT exists in {} but it should.'.format(
                            model_name, img_folder.similarity_metrics_csv
                        ))

                    model_names_hashed[model_name] = False

    

    print('\n'*5)

    print('This the ideal structure of the project directory:\n')

    print(root.content())



    










