import argparse
import warnings
warnings.filterwarnings("ignore")

#OZGUR IS HERE

import torch
import torch.optim as optim
torch.backends.cudnn.enabled       = True
torch.backends.cudnn.benchmark     = True
torch.backends.cudnn.deterministic = True

import utils.funcs as fn
import utils.basic_utils as bu
import utils.image_utils as imu
import utils.model_utils as mu
import utils.array_utils as au
import utils.denoising_utils as du
import utils.inpainting_utils as iu
from utils.common_utils import get_image_grid
from utils.gpu_utils import gpu_filter
from utils.paths import ROOT, IMG_EXT
from utils.common_types import *
from utils.keywords import *



def parse_args():
    parser = argparse.ArgumentParser(description='NAS-DIP Denoising')

    parser.add_argument('--gpu_index', default=None, type=int)
    parser.add_argument('--num_gpu', type=int, default=12)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--check', action='store_true')

    parser.add_argument('img_stem', type=str)

    parser.add_argument('--p', default=50, type=int)
    parser.add_argument('--exp_weight', default=0.99, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--reg_noise_std', default=1./30., type=float)

    parser.add_argument('--num_iter', default=4000, type=int)
    parser.add_argument('--atleast', type=int, default=500)
    parser.add_argument('--show_every', default=1, type=int)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    GPU_INDEX = args.gpu_index
    NUM_GPU = args.num_gpu
    CPU = args.cpu
    DTYPE = torch.FloatTensor if CPU else torch.cuda.FloatTensor
    CHECK = args.check

    IMG_STEM = args.img_stem
    IMG_NAME = f'{IMG_STEM}{IMG_EXT}'

    P: int = args.p # this is for images with pixel values in the range [0, 255]
    EXP_WEIGHT = args.exp_weight
    LR = args.lr
    REG_NOISE_STD = args.reg_noise_std

    NUM_ITER = args.num_iter
    ATLEAST = args.atleast
    SHOW_EVERY = args.show_every


    # stem is the name of a file without its extension
    img_name = IMG_STEM + IMG_EXT


    # load the image
    img_true_np = bu.read_true_image(INPAINTING, IMG_STEM)
    img_noisy_np, mask_np = bu.read_noisy_image(IMG_STEM, p=P, ret_noise=True)
    
    img_true_np_psd_db_norm = fn.psd_db_norm(img_true_np)
    img_noisy_np_psd_db_norm = fn.psd_db_norm(img_noisy_np)
    
    img_true_torch = imu.np_to_torch(img_true_np).type(DTYPE)
    img_noisy_torch = imu.np_to_torch(img_noisy_np).type(DTYPE)
    mask_torch = au.np_to_torch(mask_np).type(DTYPE)
    
    psnr_noisy = fn.psnr(img_true_np, img_noisy_np)
    out_channels = img_true_np.shape[0]

    print(f'Image {img_name} is loaded.')
    print(f'Shape: {img_true_np.shape}.')
    print(f'PSNR of the noisy image: {psnr_noisy:.2f} dB.')
    print()


    # we will use the same input noise on all models
    input_noise = du.get_noise_like(img_true_torch, 1/10, 'uniform').detach()
    input_noise_np = imu.torch_to_np(input_noise)
    input_noise_np_psd_db_norm = fn.psd_db_norm(input_noise_np)
    in_channels = input_noise_np.shape[0]

    print(f'input noise shape: {input_noise.shape}.')
    print()


    # read the models
    model_names = gpu_filter(GPU_INDEX, NUM_GPU)
    num_models = len(model_names)
    print(f'{num_models} models will be processed.\n')

    
    # we will save the results here
    datadir = ROOT[BENCHMARK][INPAINTING][P][IMG_STEM]

    # start to train the models
    print(f'Starting the DIP process...')
    for i, model_name in enumerate(model_names, start=1):
        print('{:03}/{:03}: {}'.format(i, len(model_names), model_name))

        # we will save the results here
        modeldir = datadir[DATA][model_name]

        # check whether the necessary files allready exists
        if CHECK and \
            modeldir['htr.pkl'].exists() and \
            modeldir['grid.png'].exists() and \
            modeldir['img_noisy.npy'].exists() and \
            modeldir['input_noise.npy'].exists() and \
            modeldir['psnr_noisy.pkl'].exists():
            print('Necessary files already exists - skipped.\n')
            continue

        # create the model
        model = mu.create_model(
            model_name, in_channels=in_channels, out_channels=out_channels
        ).type(DTYPE)
        print('Model is created.')
        print('Starting optimization with ADAM.')

        optimizer = optim.Adam(model.parameters(), lr=LR)

        # denoising
        htr = iu.inpainting(
            model=model,
            optimizer=optimizer,
            img_true_np=img_true_np,
            img_true_torch=img_true_torch,
            mask_torch=mask_torch,
            input_noise=input_noise,
            num_iter=NUM_ITER,
            atleast=ATLEAST,
            exp_weight=EXP_WEIGHT,
            reg_noise_std=REG_NOISE_STD,
            show_every=SHOW_EVERY
        )
        
        grid = get_image_grid(
            [
                input_noise_np, 
                img_true_np, 
                img_noisy_np, 
                htr['best_out'], 
                htr['best_out_sm'], 

                input_noise_np_psd_db_norm, 
                img_true_np_psd_db_norm, 
                img_noisy_np_psd_db_norm, 
                fn.psd_db_norm(htr['best_out']), 
                fn.psd_db_norm(htr['best_out_sm'])
            ], 
            nrow=5
        )

        # save the results
        modeldir['htr.pkl'].save(htr)
        modeldir['img_noisy.npy'].save(img_noisy_np)
        modeldir['input_noise.npy'].save(input_noise_np)
        modeldir['psnr_noisy.pkl'].save(psnr_noisy)
        modeldir['grid.png'].save(grid)

        print('Results are saved.\n')


if __name__ == '__main__':
    main()

