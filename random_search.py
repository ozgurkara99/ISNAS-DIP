import argparse
from collections import defaultdict
from typing import Dict

import torch
from torch.optim import Adam

import utils.array_utils as au
import utils.model_utils as mu
import utils.image_utils as imu
import utils.basic_utils as bu
import utils.funcs as fn
import utils.denoising_utils as du
import utils.inpainting_utils as iu
import utils.sr_utils as su
import utils.selection as sel
import models.downsampler as ds


from utils.gpu_utils import gpu_filter
from utils.paths import IMG_EXT
from utils.paths import ROOT
from utils.keywords import *
from utils.common_types import *

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Random search (almost) without training.')

    parser.add_argument('--gpu_index', dest='gpu_index', type=int, default=None)
    parser.add_argument('--num_gpu', dest='num_gpu', type=int, default=12)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('img_stem', type=str)

    parser.add_argument('--sigma', default=None, type=int)
    parser.add_argument('--p', default=None, type=int)
    parser.add_argument('--zoom', default=None, type=int)
    
    parser.add_argument('--exp_weight', default=0.99, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--reg_noise_std', default=1./30., type=float)

    parser.add_argument('--show_every', default=1, type=int)
    parser.add_argument('--not_verbose', action='store_true')
    
    parser.add_argument('--num_iter', default=10_000, type=int)
    parser.add_argument('--save_out_at', default='1500', type=str)
    
    parser.add_argument('--num_models', default=9999, type=int)
    parser.add_argument('--check', action='store_true')
    
    parser.add_argument('--small', action='store_true')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    GPU_INDEX = args.gpu_index
    NUM_GPU = args.num_gpu
    CPU = args.cpu
    DTYPE = torch.FloatTensor if CPU else torch.cuda.FloatTensor
    
    IMG_STEM = args.img_stem
    #PROCESS = args.process

    SIGMA = args.sigma # this is for images with pixel values in the range [0, 255]
    P = args.p # percent probability
    ZOOM = args.zoom # percent probability
    PROCESS = bu.get_process(SIGMA, P, ZOOM)
    
    EXP_WEIGHT = args.exp_weight
    LR = args.lr
    REG_NOISE_STD = args.reg_noise_std
    
    NUM_ITER = args.num_iter
    SAVE_OUT_AT = list(map(int, args.save_out_at.split(',')))
    
    CHECK = args.check
    
    SMALL = args.small
    
    
    # display the GPU related infoormation
    print('GPU index: {} Number of GPU\'s: {}'.format(GPU_INDEX, NUM_GPU))
    
    
    # stem is the name of a file without its extension
    img_name = IMG_STEM + IMG_EXT


    # load the images
    img_true_np = bu.read_true_image(PROCESS, IMG_STEM)
    img_noisy_np, noise_np = bu.read_noisy_image(IMG_STEM, SIGMA, P, ZOOM, ret_noise=True)
    
    img_true_np_orig = np.array(img_true_np)
    img_noisy_np_orig = np.array(img_noisy_np)
    if PROCESS == INPAINTING:
        noise_np_orig = np.array(noise_np)
    
    img_true_torch_orig = imu.np_to_torch(img_true_np_orig).type(DTYPE)
    img_noisy_torch_orig = imu.np_to_torch(img_noisy_np_orig).type(DTYPE)
    if PROCESS == INPAINTING:
        noise_torch_orig = au.np_to_torch(noise_np_orig).type(DTYPE)
    
    in_channels = img_true_np.shape[0]
    out_channels = img_true_np.shape[0]
    img_true_size = (img_true_np.shape[-2], img_true_np.shape[-1])
    img_noisy_size = (img_noisy_np.shape[-2], img_noisy_np.shape[-1])
    
    if PROCESS == SR and out_channels == 3:
        ycbcr = True
    else:
        ycbcr = False
    
    # create the metric maps
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
    

    print(f'Image {img_name} is loaded.')
    print(f'Shape: {img_true_np.shape}.')
    
    if PROCESS in (DENOISING, INPAINTING):
        psnr_noisy = fn.psnr(img_true_np, img_noisy_np)
        print(f'PSNR of the noisy image: {psnr_noisy:.2f} dB.')
    elif PROCESS == SR:
        print(f'Shape of the noisy image: {img_noisy_np.shape}.')
    print()
    
    
    # we will save the results here
    folder = ROOT[RANDOM_SEARCH][PROCESS]
    if PROCESS == DENOISING:
        folder = folder[SIGMA][IMG_STEM]
    elif PROCESS == INPAINTING:
        folder = folder[P][IMG_STEM]
    elif PROCESS == SR:
        folder = folder[ZOOM][IMG_STEM]
    # if SMALL:
    #     folder = folder['small']

    # read the models
    total_model_names = ROOT[RANDOM_SEARCH][MODELS_GENERATED_LST].load()
    model_names = gpu_filter(GPU_INDEX, NUM_GPU, total_model_names)
    num_models = len(model_names)
    print('{} models will be processed.\n'.format(num_models))
    
    # downsampler for the SR case
    if PROCESS == SR:
        downsampler = su.get_downsampler(ZOOM, in_channels
        ).type(DTYPE)
    
    # cache for fast calculations
    cache = fn.Cache()
    cache.register(img_noisy_np)
    
    similarity_metrics = [
        #'psd db mse',
        'psd db strip mse',
        'psd strip hist emd'
    ]
    lowpass_metrics = [
        #'psd 99_per_bw'
    ]
    other_metrics = [
        'random mse'
    ]
    metrics = similarity_metrics + lowpass_metrics + other_metrics
    similarity_metric_funcs = [Metric(metric, cache) for metric in similarity_metrics]
    lowpass_metric_funcs = [Transformation(metric, cache) for metric in lowpass_metrics]
    other_metric_funcs = [Metric(metric, cache) for metric in other_metrics]
    metric_funcs = similarity_metric_funcs + lowpass_metric_funcs + other_metric_funcs
    
    # we will store the metric results of all the 5000 models here
    metric_values = defaultdict(list)
    
    # calculate the metrics
    print('Calculating the metrics: ')
    for i, model_name in tqdm(enumerate(model_names), total=num_models):
        if i == args.num_models:
            break
        
        try:
            model = mu.create_model(model_name, in_channels, out_channels).type(DTYPE)
        except:
            with open('ERRORS.txt', 'a') as f:
                f.write(f'{model_name} - {PROCESS} - cannot create\n')
            continue
        
        
        with torch.no_grad():
            # input_noise = du.get_noise_like(img_true_torch, sigma=1/10, noise_fmt='uniform')
            # out = model(input_noise).detach()
            try:
                input_noise = du.get_noise_like(img_true_torch_orig, sigma=1/10, noise_fmt='uniform')
                out = model(input_noise).detach()
            except:
                with open('ERRORS.txt', 'a') as f:
                    f.write(f'{model_name} - {PROCESS} - other error\n')
                continue
            
            if PROCESS == SR:
                # downsampler = downsampler.type(out.dtype)
                out = downsampler(out)
            
            out = out.cpu().numpy()
        
        del model
        del input_noise

        # calculate the metrics
        metric_values['model name'].append(model_name)
        with cache.register(out):
            for metric in similarity_metric_funcs + other_metric_funcs:
                value = metric(img_noisy_np, out)
                metric_values[metric.metric].append(value)
            
            for metric in lowpass_metric_funcs:
                value = metric(out)
                metric_values[metric.transformation].append(value)
    
    # save the calculated metric values just in case
    print('Saving the results of the metric calculations.')
    metric_values = pd.DataFrame.from_dict(metric_values)
    metric_values = metric_values.set_index('model name')
    folder['metric_results.csv'].save(metric_values)
    
    
    if SMALL:
        new_img_true_size = np.array(img_true_np_orig.shape[1:])
        new_img_true_size = new_img_true_size / new_img_true_size.min() * 64
        new_img_true_size = new_img_true_size.astype(np.int32)
        
        img_true_np = imu.resize(img_true_np, new_img_true_size)
        img_noisy_np = imu.resize(img_noisy_np, new_img_true_size)
        if PROCESS == INPAINTING:
            noise_np = np.expand_dims(noise_np,axis=0)
            noise_np = imu.resize(noise_np, new_img_true_size)
            noise_np = np.squeeze(noise_np,axis=0)

    img_true_torch = imu.np_to_torch(img_true_np).type(DTYPE)
    img_noisy_torch = imu.np_to_torch(img_noisy_np).type(DTYPE)
    if PROCESS == INPAINTING:
        noise_torch = au.np_to_torch(noise_np).type(DTYPE)
    
    # now train the best 15 models for each metrics
    input_noise = du.get_noise_like(img_true_torch, sigma=1/10, noise_fmt='uniform')
    input_noise_orig = du.get_noise_like(img_true_torch_orig, sigma=1/10, noise_fmt='uniform')
    print('Training the chosen models: ')
    for metric in metrics:
        print(f'{metric}: ', end='')
        
        tmp_folder = folder[metric]
        fname = 'chosen_models.csv'
        file = tmp_folder[fname]
        
        if CHECK and file.exists():
            print('- skipped.')
            continue
        print()
        

        model_metrics = metric_values.sort_values(by=metric)
        model_metrics = model_metrics.iloc[:15]
        chosen_models = list(model_metrics.index)
        
        # train chosen models
        model_metrics = {name: model_metrics.loc[name][metric] for name in chosen_models}
        model_outputs: Dict[str, NumpyArray] = {}
        model_best_iters: Dict[str, int] = {}
        performances = defaultdict(list)
        for i, model_name in enumerate(chosen_models, start=1):
            print(f'{i:02}/{len(chosen_models)} - {model_name}:')
            
            performances['model name'].append(model_name)
            
            model = mu.create_model(model_name, in_channels, out_channels).type(DTYPE)
            metric_value = metric_values.loc[model_name][metric]
            
            if PROCESS == DENOISING: 
                optimizer = Adam(model.parameters(), lr=LR)
                htr: du.HtrDict = du.denoising(
                    model=model,
                    optimizer=optimizer,
                    img_true_np=img_true_np,
                    img_noisy_torch=img_noisy_torch,
                    input_noise=input_noise,
                    num_iter=NUM_ITER,
                    exp_weight=EXP_WEIGHT,
                    reg_noise_std=REG_NOISE_STD,
                    # get_outputs_at=SAVE_OUT_AT
                )
            elif PROCESS == INPAINTING:
                optimizer = Adam(model.parameters(), lr=LR)
                htr: iu.HtrDict = iu.inpainting(
                    model=model,
                    optimizer=optimizer,
                    img_true_np=img_true_np,
                    img_true_torch=img_true_torch,
                    mask_torch=noise_torch, 
                    input_noise=input_noise,
                    num_iter=NUM_ITER,
                    exp_weight=EXP_WEIGHT,
                    reg_noise_std=REG_NOISE_STD,
                    # get_outputs_at=SAVE_OUT_AT
                )
            elif PROCESS == SR:
                optimizer = Adam(model.parameters(), lr=LR)
                htr: su.HtrDict = su.sr(
                    model=model,
                    optimizer=optimizer,
                    img_true_np=img_true_np,
                    img_noisy_torch=img_noisy_torch,
                    input_noise=input_noise,
                    downsampler=downsampler,
                    num_iter=NUM_ITER,
                    exp_weight=EXP_WEIGHT,
                    reg_noise_std=REG_NOISE_STD,
                    # get_outputs_at=SAVE_OUT_AT
                )
                    
            # save the results
            psnr_gt_sm = htr['psnr_gt_sm']
            last_psnr_smooth = psnr_gt_sm[-1]
            best_psnr_smooth = htr['best_psnr_gt_sm']
            
            last_out_sm = htr['last_out_sm']
            best_out_sm = htr['best_out_sm']
            
            best_iter = htr['best_iter_sm']
            
            tmp_folder[model_name]['htr.pkl'].save(htr)
            tmp_folder[model_name]['out_best.png'].save(best_out_sm)
            tmp_folder[model_name]['out_last.png'].save(last_out_sm)
            for iter, out in htr['outs_sm'].items():
                tmp_folder[model_name][f'out_{iter}.png'].save(out)
            tmp_folder[model_name]['psnr_gt_sm.npy'].save(psnr_gt_sm)

                
            performances['last psnr smooth'].append(last_psnr_smooth)
            performances['best psnr smooth'].append(best_psnr_smooth)
            for iter in htr['outs'].keys():
                performances[f'{iter} psnr smooth'].append(psnr_gt_sm[iter])
            performances['best iteration'].append(best_iter)
            performances[metric].append(metric_value)
            
            model_best_iters[model_name] = best_iter
            
            model_outputs[model_name] = last_out_sm
            
        performances = pd.DataFrame.from_dict(performances)
        performances = performances.set_index('model name')
        file.save(performances)
        
        # select a model
        tmp = tmp_folder['selected_model']
        
        selected_model = sel.closest_to_average(model_outputs, model_metrics, ycbcr=ycbcr)
        
        # denoising image using this selected model
        
        if PROCESS == DENOISING: 
            optimizer = Adam(model.parameters(), lr=LR)
            htr: du.HtrDict = du.denoising(
                model=model,
                optimizer=optimizer,
                img_true_np=img_true_np_orig,
                img_noisy_torch=img_noisy_torch_orig,
                input_noise=input_noise_orig,
                num_iter=NUM_ITER,
                exp_weight=EXP_WEIGHT,
                reg_noise_std=REG_NOISE_STD,
                get_outputs_at=SAVE_OUT_AT
            )
        elif PROCESS == INPAINTING:
            optimizer = Adam(model.parameters(), lr=LR)
            htr: iu.HtrDict = iu.inpainting(
                model=model,
                optimizer=optimizer,
                img_true_np=img_true_np_orig,
                img_true_torch=img_true_torch_orig,
                mask_torch=noise_torch_orig, 
                input_noise=input_noise_orig,
                num_iter=NUM_ITER,
                exp_weight=EXP_WEIGHT,
                reg_noise_std=REG_NOISE_STD,
                get_outputs_at=SAVE_OUT_AT
            )
        elif PROCESS == SR:
            optimizer = Adam(model.parameters(), lr=LR)
            htr: su.HtrDict = su.sr(
                model=model,
                optimizer=optimizer,
                img_true_np=img_true_np_orig,
                img_noisy_torch=img_noisy_torch_orig,
                input_noise=input_noise_orig,
                downsampler=downsampler,
                num_iter=NUM_ITER,
                exp_weight=EXP_WEIGHT,
                reg_noise_std=REG_NOISE_STD,
                get_outputs_at=SAVE_OUT_AT
            )
        
        tmp[HTR_PKL].save(htr)
        last_out = htr['last_out_sm']
        last_out_psnr = fn.psnr(img_true_np_orig, last_out, ycbcr=ycbcr)
        tmp['last_out.png'].save(last_out)
        tmp['last_out_psnr.txt'].save(f'{last_out_psnr}')
        
        for iter, out in htr['outs'].items():
            out_psnr = fn.psnr(img_true_np_orig, out, ycbcr=ycbcr)
            tmp[f'{iter}_out.png'].save(out)
            tmp[f'{iter}_out_psnr.txt'].save(f'{out_psnr}')
        
        tmp['selected_model.txt'].save(selected_model)
            
            
        
    print('DONE!')


if __name__ == '__main__':
    main()
