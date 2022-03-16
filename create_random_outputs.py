import argparse
import warnings
warnings.filterwarnings("ignore")

import torch

import numpy as np

from models.cross_skip import skip
from utils.common_utils import get_noise

from utils.basic_utils import load_obj
from utils.gpu_utils import gpu_filter
from utils.paths import NUM_RANDOM_OUTPUTS as _NUM_RANDOM_OUTPUTS
from utils.paths import IMG_SIZE
from utils.paths import root



def get_model(model_name):
    temp = model_name.split('_')

    index = temp[1]
    skip_connect = temp[-1]

    index = int(index)

    skip_connect = list(map(int, skip_connect))
    skip_connect = np.array(skip_connect)
    skip_connect = np.reshape(skip_connect, (5, 5))

    in_channels = 1
    out_channels = 1
    
    model = skip(
        model_index=index,
        skip_index=skip_connect,
        num_input_channels=in_channels,
        num_output_channels=out_channels,
        num_channels_down=[128] * 5,
        num_channels_up=[128] * 5,
        num_channels_skip=[4] * 5,
        upsample_mode='bilinear',
        downsample_mode='stride',
        need_sigmoid=True,
        need_bias=True,
        pad='reflection',
        act_fun='LeakyReLU'
    )

    return model



# read the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num', default=_NUM_RANDOM_OUTPUTS, type=int)
parser.add_argument('--gpu_index', type=int, default=None)
parser.add_argument('--num_gpu', type=int, default=12)
parser.add_argument('--mangling', action='store_true')
parser.add_argument('--check', action='store_true')
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

GPU_INDEX = args.gpu_index
NUM_GPU = args.num_gpu
MANGLING = args.mangling
CHECK = args.check
CPU = args.cpu
DTYPE = torch.FloatTensor if CPU else torch.cuda.FloatTensor
NUM_RANDOM_OUTPUTS = args.num



# display the GPU related infoormation
print('GPU index: {} Number of GPU\'s: {}'.format(GPU_INDEX, NUM_GPU))



# read the models
model_names = gpu_filter(GPU_INDEX, NUM_GPU, mangling=MANGLING)
num_models = len(model_names)
print('{} models will be processed.\n'.format(num_models))



# create the random outputs
for i, model_name in enumerate(model_names, start=1):
    print('{:03}/{:03}: {}'.format(i, num_models, model_name))
    
    with torch.no_grad():
        model = get_model(model_name).type(DTYPE)
        print('Model is created.')

        # we will save the random outputs into this directory
        data_dir = root.benchmark.random_outputs[model_name]


        print('Creating the outputs:')
        for j in range(NUM_RANDOM_OUTPUTS):
            fname = 'random_output_{:04}.npy'.format(j)
            file = data_dir[fname]

            print('    {:03}/{:03}: {} - '.format(
                j+1, NUM_RANDOM_OUTPUTS, file.name
            ), end='')


            # check if this random output is already exists
            if CHECK and file.exists():
                print('already exists - skipped.')
                continue

            
            # (1, 1, IMG_SIZE, IMG_SIZE)
            input_noise = get_noise(1, 'noise', (IMG_SIZE, IMG_SIZE)).type(DTYPE)
            random_output = model(input_noise).detach().cpu().numpy()
            print('created - ', end='')


            # save to the file
            file.save(random_output, overwrite=True)

            print('saved.')

    print()
                            

print('\nAll outputs are created.\n')

