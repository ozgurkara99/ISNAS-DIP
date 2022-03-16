import argparse

from tqdm import tqdm
import utils.image_utils as imu
import utils.basic_utils as bu
from utils.paths import ROOT
from utils.keywords import *

def parse_args():
    parser = argparse.ArgumentParser(description='Random search (almost) without training.')

    # parser.add_argument('--gpu_index', dest='gpu_index', type=int, default=None)
    # parser.add_argument('--num_gpu', dest='num_gpu', type=int, default=12)

    # parser.add_argument('img_stem', type=str)
    # parser.add_argument('--process', type=str, default=DENOISING)

    parser.add_argument('--sigma', default=None, type=int)
    parser.add_argument('--p', default=None, type=int)
    parser.add_argument('--zoom', default=None, type=int)

    parser.add_argument('--check', action='store_true')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    SIGMA = args.sigma
    P = args.p
    ZOOM = args.zoom
    
    for PROCESS in (DENOISING, INPAINTING, SR):
        
        if PROCESS == DENOISING and SIGMA is None:
            continue
        if PROCESS == INPAINTING and P is None:
            continue
        if PROCESS == SR and ZOOM is None:
            continue
        
        # make the true images divisible by 32
        print('Making images divisible by 32.')
        imgs = bu.read_true_images(PROCESS)
        for stem, img in imgs.items():
            name = bu.name_from_stem(stem)
            img = imu.crop_image(img, 32)
            ROOT[IMAGES][PROCESS][TRUE][name].save(img)
        
        # create the noisy images
        if PROCESS == DENOISING:
            bu.read_noisy_images(PROCESS, sigma=SIGMA)
        if PROCESS == INPAINTING:
            bu.read_noisy_images(PROCESS, p=P)
        if PROCESS == SR:
            bu.read_noisy_images(PROCESS, zoom=ZOOM)
        
            
if __name__ == '__main__':
    main()

