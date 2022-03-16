import contextlib
import argparse
from collections import defaultdict
from time import sleep
from pprint import pprint

import pandas as pd
from models.downsampler import Downsampler

import utils.funcs as fn
import utils.basic_utils as bu
import utils.array_utils as au
import utils.sr_utils as su
import utils.metric_utils as metu

from utils.gpu_utils import gpu_filter
from utils.paths import IMG_EXT
from utils.paths import ROOT
from utils.common_types import *
from utils.keywords import *


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('img_stem', type=str)
    parser.add_argument('--process', type=str, default=DENOISING)
    parser.add_argument('--sigma', type=int, default=25)
    parser.add_argument('--p', type=int, default=50)
    parser.add_argument('--zoom', type=int, default=4)
    
    parser.add_argument('--gpu_index', type=int, default=None)
    parser.add_argument('--num_gpu', type=int, default=12)
    parser.add_argument('--wait', type=float, default=20)
    
    parser.add_argument('--overwrite', action='store_true')
    
    # parser.add_argument('--high_lr', action='store_true')
    # parser.add_argument('--noisy', action='store_true')
    
    parser.add_argument('--read_psnr_from_csv', action='store_true')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    GPU_INDEX = args.gpu_index
    NUM_GPU = args.num_gpu
    WAIT = args.wait
    
    OVERWRITE = args.overwrite
    
    # HIGH_LR = args.high_lr
    # NOISY = args.noisy
    
    READ_PSNR_FROM_CSV = args.read_psnr_from_csv

    IMG_STEM = args.img_stem
    img_name = IMG_STEM + IMG_EXT
    
    PROCESS = args.process
    SIGMA = args.sigma
    P = args.p
    ZOOM = args.zoom


    # display the GPU related information
    print('GPU index: {} Number of GPU\'s: {}'.format(GPU_INDEX, NUM_GPU))

    # read the models
    model_names = gpu_filter(GPU_INDEX, NUM_GPU)
    num_models = len(model_names)
    print('{} models will be processed.'.format(num_models))


    # load the image
    print('Image {} is being loaded...'.format(img_name), end='')
    
    img_true_np = bu.read_true_image(PROCESS, IMG_STEM)
    
    if PROCESS == DENOISING:
        img_noisy_np = bu.read_noisy_image(PROCESS, IMG_STEM, sigma=SIGMA)
        psnr_noisy = fn.psnr(img_true_np, img_noisy_np)
    if PROCESS == INPAINTING:
        img_noisy_np, mask_np = bu.read_noisy_image(PROCESS, IMG_STEM, p=P, ret_noise=True)
        psnr_noisy = fn.psnr(img_true_np, img_noisy_np)
    if PROCESS == SR:
        img_noisy_np = bu.read_noisy_image(PROCESS, IMG_STEM, zoom=ZOOM)
        downsampler = su.get_downsampler(ZOOM).cpu()
    
    img_true_size = (img_true_np.shape[1], img_true_np.shape[2])
    img_noisy_size = (img_noisy_np.shape[1], img_noisy_np.shape[2])
    
    print(' - loaded.')
    print('Shape: {}\n'.format(img_true_np.shape))
    
    
    # create metric maps
    maps = fn.UsefullMaps(img_size=img_noisy_size)
    class Transformation(fn.Transformation):
        def __init__(self, transformation: str, cache: fn.Cache) -> None:
            super().__init__(
                transformation, maps.transformation_map, cache
            )

    class Metric(fn.Metric):
        def __init__(self, metric: str, cache: fn.Cache) -> None:
            super().__init__(
                metric, 
                maps.transformation_map,
                maps.loss_map,
                cache
            )


    # read the existing csv file
    if PROCESS == DENOISING:
        csv = ROOT[BENCHMARK][PROCESS][SIGMA][IMG_STEM]['similarity_metrics_noisy_img.csv']
    if PROCESS == INPAINTING:
        csv = ROOT[BENCHMARK][PROCESS][P][IMG_STEM]['similarity_metrics_noisy_img.csv']
    if PROCESS == SR:
        csv = ROOT[BENCHMARK][PROCESS][ZOOM][IMG_STEM]['similarity_metrics_noisy_img.csv']

    # this is the old csv file
    old_df = None
    if csv.exists():
        if not OVERWRITE:
            old_df = csv.load()
    # to make sure that all processes of this script reads this csv file, sleep 
    # before deleting it.
    sleep(WAIT)
    with contextlib.suppress(FileNotFoundError):
        csv.delete()
        
    
    # for fast calculations, we will store middle steps in a cache
    cache = fn.Cache()
    # do not forget to unregister afterwards
    cache.register(img_noisy_np)
        
    # these are the metrics to be calculated
    # read the metrics from the yaml file
    # yaml_file = ROOT[SIMILARITY_METRICS_YAML]
    # yaml_obj = yaml_file.load()
    # metrics = yaml_obj['noisy image']['low lr']
    metrics = metu.get_similarity_metrics()
            
    # metrics = [
    #     'psd db mse',
    #     'psd db strip mse',
        
    #     'nodc psd strip mse',
    #     'nodc psd db mse',
    #     'nodc psd db strip mse',
        
    #     'psd db hist emd',
    #     'nodc psd hist emd',
    #     'nodc psd db hist emd',
    #     'psd strip hist emd',
    # ]
    
    # do not recalculate the same metrics
    if not OVERWRITE and old_df is not None:
        metrics = [metric for metric in metrics if metric not in old_df.columns]
    
    print('The following metrics will be calculated: ')
    pprint(metrics)
    
    # we will store the results here
    data = defaultdict(list)
    
    # calculate metrics
    for i, model_name in enumerate(model_names, start=1):
        print('{:03}/{:03}: {}'.format(i, num_models, model_name))
        
        # this is the random outputs of the model
        out = ROOT[BENCHMARK][RANDOM_OUTPUTS][model_name][RANDOM_OUTPUT_NPY].load()
        if PROCESS == SR:
            out = bu.np_to_torch(out).cpu()
            out = downsampler(out)
            out = bu.torch_to_np(out)
            
        # load the htr object
        if PROCESS == DENOISING:
            htr = ROOT[BENCHMARK][PROCESS][SIGMA][IMG_STEM][DATA][model_name]['htr.pkl'].load()
        if PROCESS == INPAINTING:
            htr = ROOT[BENCHMARK][PROCESS][P][IMG_STEM][DATA][model_name]['htr.pkl'].load()
        if PROCESS == SR:
            htr = ROOT[BENCHMARK][PROCESS][ZOOM][IMG_STEM][DATA][model_name]['htr.pkl'].load()
        
        # store the results in a dictionary, this will be a row of the output
        # csv file
        data['model name'].append(model_name)
        
        data['best psnr smooth'].append(htr['best_psnr_gt_sm'])
        data['best iteration smooth'].append(htr['best_iter_sm'])
        
        data['best psnr'].append(htr['best_psnr_gt'])
        data['best iteration'].append(htr['best_iter'])
        
        if PROCESS in (DENOISING, INPAINTING):
            data['psnr noisy'].append(psnr_noisy)
            data['best psnr increase'].append(htr['best_psnr_gt'] - psnr_noisy)
            data['best psnr increase smooth'].append(htr['best_psnr_gt_sm'] - psnr_noisy)
            
        # now, calculate metrics
        
        # to use cache, register out array
        # cache.register(out)
        for metric in metrics:
            with cache.register(out):
                func = Metric(metric, cache)
                result = func(img_noisy_np, out)
                data[metric].append(result)
            
        # do not forget to unregister out array, because it will prevent
        # garbage collection
        # cache.unregister(out)


        # the values from the existing file is used
        if old_df is not None:
            for col in old_df.columns:
                if col in metrics:
                    continue
                
                tmp = (
                    'model name', 'best psnr smooth', 'best iteration smooth',
                    'best psnr', 'best iteration', 'psnr noisy',
                    'best psnr increase', 'best psnr increase smooth',
                    'psnr increase', 'psnr increase smooth'
                )
                if col in tmp:
                    continue
                
                data[col].append(
                    old_df.loc[model_name][col]
                )

        # display the calculated metrics
        for key in data:
            if key == 'model name':
                continue
            
            val =  data[key][-1]
            print('{:<30}: {}'.format(key, val))
        print()


    print('Calculations are finished.')


    # save the data into a csv file
    new_df = pd.DataFrame.from_dict(data)
    new_df = new_df.set_index('model name')

    print('Saving into {}...'.format(csv), end='')
    
    csv.save(new_df, append=True)

    print(' - saved.')


if __name__ == '__main__':
    main()





