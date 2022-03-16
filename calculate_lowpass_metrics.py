import argparse
from collections import defaultdict
from time import sleep
from pprint import pprint
import contextlib

import pandas as pd

import utils.funcs as fn
import utils.metric_utils as metu

from utils.gpu_utils import gpu_filter
from utils.paths import IMG_SIZE, IMG_EXT
from utils.paths import root
from utils.common_types import *
from utils.keywords import *



class Transformation(fn.Transformation):
    def __init__(self, transformation: str, cache: fn.Cache) -> None:
        super().__init__(
            transformation, fn.UsefullMaps.transformation_map, cache
        )


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu_index', dest='gpu_index', type=int, default=None)
    parser.add_argument('--num_gpu', dest='num_gpu', type=int, default=12)
    parser.add_argument('--wait', type=float, default=20)
    
    parser.add_argument('--overwrite', action='store_true')
    
    parser.add_argument('--high_lr', action='store_true')
    parser.add_argument('--img_stem', type=str)
    
    args = parser.parse_args()
    return args

def main():
    # read the command line arguments
    args = parse_args()

    GPU_INDEX = args.gpu_index
    NUM_GPU = args.num_gpu
    WAIT = args.wait
    
    OVERWRITE = args.overwrite

    HIGH_LR = args.high_lr


    # display the GPU related infoormation
    print('GPU index: {} Number of GPU\'s: {}'.format(GPU_INDEX, NUM_GPU))



    # read the models
    model_names = gpu_filter(GPU_INDEX, NUM_GPU)
    num_models = len(model_names)
    print('{} models will be processed.\n'.format(num_models))
    
    
    if HIGH_LR:
        img_stem = args.img_stem
        img_name = img_stem + IMG_EXT
        
        # load the image
        print('Image {} is being loaded...'.format(img_name), end='')

        img_file = root['images']['denoising'][img_name]
        img_true = img_file.load(format='np', size=IMG_SIZE, d=32)
        
        print(' - loaded.')
        print('Shape: {}\n'.format(img_true.shape))
    

    # read the existing csv file
    if HIGH_LR:
        csv = root[BENCHMARK][DENOISING][img_stem][LOWPASS_METRICS_HIGH_LR_CSV]
    else:
        csv = root[BENCHMARK][LOWPASS_METRICS_CSV]
    
     # this is the old csv file
    old_df = None
    if csv.exists():
        if not OVERWRITE and csv.exists():
            old_df = csv.load()
    # to make that sure all processes of this script reads this csv file, sleep 
    # before deleting it.
    sleep(WAIT)
    with contextlib.suppress(FileNotFoundError):
        csv.delete()
        
        
    
    # for fast calculations, we will store middle steps in a cache
    cache = fn.Cache()
    
    
    # these are the metrics to be calculated
    # read the metrics from the yaml file
    # yaml_file = root[LOWPASS_METRICS_YAML]
    # yaml_obj = yaml_file.load()
    # if HIGH_LR:
    #     metrics = yaml_obj['high lr']
    # else:
    #     metrics = yaml_obj['low lr']
    metrics = metu.load_lowpass_metrics()
        
    # metrics = [
        # 'psd 99_per_bw',
        # 'nodc psd 75_per_bw',
        
        # 'nodc psd db 50_db_bw',
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
        if HIGH_LR:
            out = root[BENCHMARK][DENOISING][img_stem][DATA][model_name][HIGH_LR_OUTPUT_NPY].load()
        else:
            out = root['benchmark']['random_outputs'][model_name]['random_output.npy'].load()
        
        # store the results in a dictionary, this will be a row of the output
        # csv file
        data['model name'].append(model_name)
        

        # now, calculate metrics
        
        # to use cache, register out array
        # cache.register(out)

        for metric in metrics:
            with cache.register(out):
                func = Transformation(metric, cache)
                result = func(out)
                data[metric].append(result)
            
        # do not forget to unregister out array, because it will prevent
        # garbage collection
        # cache.unregister(out)
        
        
        # the values from the existing file is used
        if old_df is not None:
            for col in old_df.columns:
                if col in metrics:
                    continue
                
                if col == 'model name':
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