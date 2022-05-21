import argparse
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
# torch.backends.cudnn.enabled       = True
# torch.backends.cudnn.benchmark     = True
# torch.backends.cudnn.deterministic = True
#commit merhaba

import utils.basic_utils as bu
import utils.array_utils as au
import utils.funcs as fn
import utils.denoising_utils as du
import utils.inpainting_utils as iu
import utils.sr_utils as su
import utils.image_utils as imu
import utils.model_utils as mu
import utils.metric_utils as metu
from models import model_denoising, model_inpainting, model_sr
from dip import dip
from utils.common_utils import get_image_grid
from utils.paths import ROOT
from utils.keywords import *
from utils.common_types import *



def parse_args():
    parser = argparse.ArgumentParser(description='NAS-DIP Denoising')

    parser.add_argument('--gpu_index', default=0, type=int)
    parser.add_argument('--num_gpu', type=int, default=12)
    parser.add_argument('--mangling', action='store_true')
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('img_stem', type=str)
    
    parser.add_argument('--in_channels', default=32, type=int)

    parser.add_argument('--sigma', default=None, type=int)
    parser.add_argument('--p', default=None, type=int)
    parser.add_argument('--zoom', default=None, type=int)
    
    parser.add_argument('--exp_weight', default=0.99, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--reg_noise_std', default=1./30., type=float)

    parser.add_argument('--num_iter', default=10_000, type=int)
    parser.add_argument('--atleast', type=int, default=None)
    
    parser.add_argument('--check', action='store_true')

    parser.add_argument('--repeat', default=15, type=int)
    
    parser.add_argument('--save_out_at', type=str, default=1200)

    args = parser.parse_args()
    return args

def get_nasdip_model_denoising(
    out_channels: int, in_channels: int = 32
) -> Model:
    
    model = model_denoising.Model(
        num_input_channels=in_channels,
        num_output_channels=out_channels
    )
    
    return model
    
def get_nasdip_model_inpainting(
    out_channels: int, in_channels: int = 32
) -> Model:
    
    model = model_inpainting.Model(
        num_input_channels=in_channels,
        num_output_channels=out_channels
    )
    
    return model

def get_nasdip_model_sr(
    out_channels: int, in_channels: int = 32
) -> Model:
    
    model = model_sr.Model(
        num_input_channels=in_channels,
        num_output_channels=out_channels
    )
    
    return model

def get_nasdip_model(
    process: Process, out_channels: int, in_channels: int = 32
) -> Model:

    if process == DENOISING:
        model = get_nasdip_model_denoising(out_channels, in_channels)
        return model
    if process == INPAINTING:
        model = get_nasdip_model_inpainting(out_channels, in_channels)
        return model
    if process == SR:
        model = get_nasdip_model_sr(out_channels, in_channels)
        return model
    
    raise TypeError(f'process should be one of {DENOISING}, {INPAINTING} or {SR}')


# def nasdip_performance(
    # repeat: int,
    # model: Model, 
    # img_stem: str, 
    # input_noise_torch: Tensor, 
    # dtype,
    # optimizer: optim.Optimizer,
    # sigma: int = None, p: int = None, zoom: int = None, 
    # downsampler: Optional[Model] = None,
    # num_iter: int = 10_000, atleast: Optional[int] = None,
    # exp_weight: float = 0.99, lr: float = 1e-2, reg_noise_std: float = 1/30,
    # save: bool = False,
    # save_out_at: Optional[Union[int, List[int]]] = None
# ) -> Tuple[DataFrame, HtrDict]:
    
    # # decide the process according to the given parameters
    # process = bu.get_process(sigma=sigma, p=p, zoom=zoom)
    
    # # load the images
    # img_true_np = bu.read_true_image(process, img_stem)
    # img_true_np_psd_db_norm = fn.psd_db_norm(img_true_np)
    # img_true_torch = imu.np_to_torch(img_true_np).type(dtype)
    # img_true_in_channels = img_true_np.shape[0]
    # if process == DENOISING:
    #     img_noisy_np = bu.read_noisy_image(process, img_stem, sigma=sigma)
    # elif process == INPAINTING:
    #     img_noisy_np, mask_np = bu.read_noisy_image(
    #         process, img_stem, p=p, ret_noise=True
    #     )
    #     mask_torch = au.np_to_torch(mask_np).type(dtype)
    # elif process == SR:
    #     img_noisy_np = bu.read_noisy_image(process, img_stem, zoom=zoom)
        
    #     if downsampler is None:
    #         downsampler = su.get_downsampler(
    #             zoom=zoom, in_channels=img_true_in_channels
    #         ).type(dtype)
    # img_noisy_torch = imu.np_to_torch(img_noisy_np).type(dtype)
    # # img_noisy_np_psd_db_norm = fn.psd_db_norm(img_noisy_np)
    
    # # get the random output of the model
    # random_output = mu.get_random_output(
    #     model=model,
    #     input_noise=input_noise_torch
    # )
    # if process == SR:
    #     tmp = imu.np_to_torch(random_output).type(dtype)
    #     tmp = downsampler(tmp)
    #     random_output = imu.torch_to_np(tmp)
    # # get the metrics
    # sim_metrics = metu.get_similarity_metrics()
    # low_metrics = metu.get_lowpass_metrics()
    # # we will store the metric results here
    # metric_results: Dict[str, list] = defaultdict(list)
    # # calculate the metrics
    # for metric in sim_metrics + low_metrics:
    #     result = metu.calculate(metric, random_output, img_noisy_np)
    #     metric_results[metric].append(result)
    
    # # train the model
    # if process == DENOISING:
    #     htr: HtrDict = du.denoising(
    #         model=model, 
    #         optimizer=optimizer, 
    #         img_true_np=img_true_np,
    #         img_noisy_torch=img_noisy_torch,
    #         input_noise=input_noise_torch,
    #         num_iter=num_iter,
    #         atleast=atleast,
    #         exp_weight=exp_weight,
    #         reg_noise_std=reg_noise_std,
    #         get_outputs=save_out_at        )
        
    #     metric_results['best psnr smooth'] = htr['best_psnr_gt_sm']
    #     metric_results['best psnr'] = htr['best_psnr_gt']
    #     metric_results['best iteration smooth'] = htr['best_iter_sm']
    #     metric_results['best iteration'] = htr['best_iter']
    #     metric_results = pd.DataFrame.from_dict(metric_results)
        
    #     grid = get_image_grid(
    #         [
    #             img_true_np, 
    #             htr['best_out'], 
    #             htr['best_out_sm'], 

    #             img_true_np_psd_db_norm, 
    #             fn.psd_db_norm(htr['best_out']), 
    #             fn.psd_db_norm(htr['best_out_sm'])
    #         ], 
    #         nrow=3
    #     )
        
    #     if save:
    #         ROOT[BENCHMARK][process][sigma][img_stem][DATA][BEST + "_" + str(repeat)][HTR_PKL].save(htr)
    #         ROOT[BENCHMARK][process][sigma][img_stem][DATA][BEST + "_" + str(repeat)][METRICS_CSV].save(metric_results)
    #         ROOT[BENCHMARK][process][sigma][img_stem][DATA][BEST + "_" + str(repeat)][GRID_PNG].save(grid)      
    # elif process == INPAINTING:
    #     htr = iu.inpainting(
    #         model=model,
    #         optimizer=optimizer,
    #         img_true_np=img_true_np,
    #         img_true_torch=img_true_torch,
    #         mask_torch=mask_torch,
    #         input_noise=input_noise_torch,
    #         num_iter=num_iter,
    #         atleast=atleast,
    #         exp_weight=exp_weight,
    #         reg_noise_std=reg_noise_std
    #     )
        
    #     metric_results['best psnr smooth'] = htr['best_psnr_gt_sm']
    #     metric_results['best psnr'] = htr['best_psnr_gt']
    #     metric_results['best iteration smooth'] = htr['best_iter_sm']
    #     metric_results['best iteration'] = htr['best_iter']
    #     metric_results = pd.DataFrame.from_dict(metric_results)
        
    #     grid = get_image_grid(
    #         [
    #             img_true_np, 
    #             htr['best_out'], 
    #             htr['best_out_sm'], 

    #             img_true_np_psd_db_norm, 
    #             fn.psd_db_norm(htr['best_out']), 
    #             fn.psd_db_norm(htr['best_out_sm'])
    #         ], 
    #         nrow=3
    #     )
        
    #     if save:
    #         ROOT[BENCHMARK][process][p][img_stem][DATA][BEST][HTR_PKL].save(htr)
    #         ROOT[BENCHMARK][process][p][img_stem][DATA][BEST][METRICS_CSV].save(metric_results)
    #         ROOT[BENCHMARK][process][p][img_stem][DATA][BEST][GRID_PNG].save(grid)
    # elif process == SR:
    #     htr = su.sr(
    #         model=model,
    #         optimizer=optimizer,
    #         img_true_np=img_true_np,
    #         img_noisy_torch=img_noisy_torch,
    #         input_noise=input_noise_torch,
    #         downsampler=downsampler,
    #         num_iter=num_iter,
    #         atleast=atleast,
    #         exp_weight=exp_weight,
    #         reg_noise_std=reg_noise_std   
    #     )
        
    #     metric_results['best psnr smooth'] = htr['best_psnr_gt_sm']
    #     metric_results['best psnr'] = htr['best_psnr_gt']
    #     metric_results['best iteration smooth'] = htr['best_iter_sm']
    #     metric_results['best iteration'] = htr['best_iter']
    #     metric_results = pd.DataFrame.from_dict(metric_results)
        
    #     grid = get_image_grid(
    #         [
    #             img_true_np, 
    #             htr['best_out'], 
    #             htr['best_out_sm'], 

    #             img_true_np_psd_db_norm, 
    #             fn.psd_db_norm(htr['best_out']), 
    #             fn.psd_db_norm(htr['best_out_sm'])
    #         ], 
    #         nrow=3
    #     )

    #     if save:
    #         ROOT[BENCHMARK][process][zoom][img_stem][DATA][BEST + "_" + str(repeat)][HTR_PKL].save(htr)
    #         ROOT[BENCHMARK][process][zoom][img_stem][DATA][BEST + "_" + str(repeat)][METRICS_CSV].save(metric_results)
    #         ROOT[BENCHMARK][process][zoom][img_stem][DATA][BEST + "_" + str(repeat)][GRID_PNG].save(grid)
        
    # return metric_results, htr

def check(
    img_stem: str,
    sigma: Optional[int] = None,
    p: Optional[int] = None,
    zoom: Optional[int] = None
) -> bool:
    
    process = bu.get_process(sigma=sigma, p=p, zoom=zoom)
    
    res = True
    res = res and ROOT[BENCHMARK][process][sigma][img_stem][DATA][BEST][HTR_PKL].exist()
    res = res and ROOT[BENCHMARK][process][sigma][img_stem][DATA][BEST][METRICS_CSV].exis()
    return res
    

def main():
    args = parse_args()
    
    CPU = args.cpu
    DTYPE = torch.FloatTensor if CPU else torch.cuda.FloatTensor

    IMG_STEM = args.img_stem

    SIGMA = args.sigma
    P = args.p
    ZOOM = args.zoom
    
    IN_CHANNELS = args.in_channels
            
    EXP_WEIGHT = args.exp_weight
    LR = args.lr
    REG_NOISE_STD = args.reg_noise_std

    NUM_ITER = args.num_iter
    ATLEAST = args.atleast
    
    CHECK = args.check
    
    SAVE_OUT_AT = args.save_out_at
    SAVE_OUT_AT = list(map(int, SAVE_OUT_AT.split(',')))
    REPEAT = args.repeat
    
    
    # decide the process according to the given parameters
    process = bu.get_process(sigma=SIGMA, p=P, zoom=ZOOM)
    
    # save the files here
    dir = ROOT[NASDIP][process]
    if process == DENOISING:
        dir = dir[SIGMA]
    elif process == INPAINTING:
        dir = dir[P]
    elif process == SR:
        dir = dir[ZOOM]
    
    if IMG_STEM.upper() in DATASETS:
        img_stems = DATASETS[IMG_STEM]
    else:
        img_stems = [IMG_STEM]
    
    
        # start dip process
    for j, img_stem in enumerate(img_stems, start=1):
        print(f'{j}\{len(img_stems)} {img_stem}: ')
        
        # read the image
        img_true_np = bu.read_true_image(process, img_stem)
        out_channels = img_true_np.shape[0]
        
        input_noise_torch = du.get_noise(
            (1, IN_CHANNELS, img_true_np.shape[1], img_true_np.shape[2]),
            sigma=1/10, noise_fmt='uniform', arr_fmt='torch'
        ).detach().type(DTYPE)
        
        
        for i in range(REPEAT):
            model = get_nasdip_model(process, out_channels=out_channels).type(DTYPE)
            print(f'{i+1}\{REPEAT}: ')
            
            tmp = dir[img_stem][f'{i:02}']
            
            if CHECK and \
               tmp[HTR_PKL].exists() and \
               tmp[METRICS_CSV].exists() and \
               tmp[GRID_PNG].exists():
                   continue
            
            optimizer = optim.Adam(model.parameters(), lr=LR)
            
            # dip process
            htr, _, _ = dip(
                model=model,
                img_stem=img_stem,
                input_noise_torch=input_noise_torch,
                dtype=DTYPE,
                optimizer=optimizer,
                sigma=SIGMA,
                p=P,
                zoom=ZOOM,
                num_iter=NUM_ITER,
                atleast=ATLEAST,
                lr=LR,
                exp_weight=EXP_WEIGHT,
                reg_noise_std=REG_NOISE_STD,
                get_outputs_at=SAVE_OUT_AT,
                dir=tmp
            )
            
            # save the outputs
            tmp['best_out.png'].save(htr['best_out'])
            tmp['best_out_sm.png'].save(htr['best_out_sm'])
            tmp['last_out.png'].save(htr['last_out'])
            tmp['last_out_sm.png'].save(htr['last_out_sm'])
            
            for iter, out in htr['outs'].items():
                tmp[f'out_{iter}.png'].save(out)
            
            for iter, out in htr['outs_sm'].items():
                tmp[f'out_sm_{iter}.png'].save(out)
        
        print()
            
        print('DONE!')
    
if __name__ == '__main__':
    main()

