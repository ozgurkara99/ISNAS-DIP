import os
import argparse

import numpy as np

from utils.gpu_utils import gpu_filter
from utils.paths import NUM_RANDOM_OUTPUTS as _NUM_RANDOM_OUTPUTS
from utils.paths import IMG_SIZE
from utils.paths import root



# read the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num', default=_NUM_RANDOM_OUTPUTS, type=int)
parser.add_argument('--gpu_index', type=int, default=None)
parser.add_argument('--num_gpu', type=int, default=12)
parser.add_argument('--mangling', action='store_true')
parser.add_argument('--check', action='store_true')
args = parser.parse_args()

GPU_INDEX = args.gpu_index
NUM_GPU = args.num_gpu
MANGLING = args.mangling
CHECK = args.check
NUM_RANDOM_OUTPUTS = args.num



# display the GPU related infoormation
print('GPU index: {} Number of GPU\'s: {}'.format(GPU_INDEX, NUM_GPU))



# read the models
model_names = gpu_filter(GPU_INDEX, NUM_GPU, mangling=MANGLING)
num_models = len(model_names)
print('{} models will be processed.\n'.format(num_models))


# start to concatenate the random outputs
for i, model_name in enumerate(model_names, start=1):
    print('{:03}/{:03}: {}'.format(i, num_models, model_name))

    # random outputs are here
    data_dir = root.benchmark.random_outputs[model_name]

    
    # check whether the necessary file already exists
    if CHECK and data_dir.random_output_npy.exists():
        shape = data_dir.random_output_npy.shape
        if len(shape) == 4 and shape[0] == NUM_RANDOM_OUTPUTS and \
            shape[1] == 1 and shape[2] == shape[3] == IMG_SIZE:

            print('{} already exists - skipped.'.format(
                data_dir.random_output_npy.name
            ))
            continue


    # read the random outputs one by one
    random_outputs = []
    for j in range(NUM_RANDOM_OUTPUTS):
        fname = 'random_output_{:04}.npy'.format(j)
        file = data_dir[fname]

        print('{:04}/{:04}: {} - '.format(j, num_models, file.name), end='')

        random_outputs.append(
            file.load()
        )

        print('readed.')

    print()

    # create the big random output array
    random_output = np.concatenate(random_outputs, axis=0)
    
    print('{} - '.format(data_dir.random_output_npy.name))
    data_dir.random_output_npy.save(random_output)
    print('saved.\n')

print('Finished.')
