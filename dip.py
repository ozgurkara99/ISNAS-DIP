import argparse
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
# torch.backends.cudnn.enabled       = True
# torch.backends.cudnn.benchmark     = True
# torch.backends.cudnn.deterministic = True
from tqdm import trange

import utils.basic_utils as bu
import utils.array_utils as au
import utils.funcs as fn
import utils.denoising_utils as du
import utils.inpainting_utils as iu
import utils.sr_utils as su
import utils.image_utils as imu
import utils.model_utils as mu
import utils.metric_utils as metu
from models import skip, get_texture_nets, resnet, unet
from utils.common_utils import get_image_grid
from utils.paths import ROOT, Folder
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
    
    parser.add_argument('--save_out_at', type=str, default='')
    parser.add_argument('--save_out_every', type=int, default=-1)

    args = parser.parse_args()
    return args


def get_net(input_depth, NET_TYPE, pad, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
    if NET_TYPE == 'ResNet':
        # TODO
        net = resnet(input_depth, 3, 10, 16, 1, nn.BatchNorm2d, False)
    elif NET_TYPE == 'skip':
        net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

    elif NET_TYPE == 'texture_nets':
        net = get_texture_nets(inp=input_depth, ratios = [32, 16, 8, 4, 2, 1], fill_noise=False,pad=pad)

    elif NET_TYPE =='UNet':
        net = unet(num_input_channels=input_depth, num_output_channels=3, 
                   feature_scale=4, more_layers=0, concat_x=False,
                   upsample_mode=upsample_mode, pad=pad, norm_layer=nn.BatchNorm2d, need_sigmoid=True, need_bias=True)
    elif NET_TYPE == 'identity':
        assert input_depth == 3
        net = nn.Sequential()
    else:
        assert False

    return net

def get_dip_model_denoising(
    out_channels: int, in_channels: int = 32
) -> Model:
    
    model = get_net(
        in_channels, 'skip', 'reflection', n_channels=out_channels,
        skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, 
        upsample_mode='bilinear'
    )
    
    return model

def get_dip_model_inpainting(
    out_channels: int, in_channels: int = 32
) -> Model:
    
    model = skip(
        in_channels, out_channels,
        num_channels_down=[128] * 5,
        num_channels_up=[128] * 5,
        num_channels_skip=[128] * 5,
        filter_size_up=3, filter_size_down=3,
        upsample_mode='nearest', filter_skip_size=1,
        need_sigmoid=True, need_bias=True, pad='reflection',
        act_fun='LeakyReLU'
    )
    
    return model

def get_dip_model_sr(
    out_channels: int, in_channels: int = 32
) -> Model:
    
    model = get_net(
        in_channels, 'skip', 'reflection', n_channels=out_channels,
        skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5,
        upsample_mode='bilinear'
    )
    
    return model

def get_dip_model(
    process: Process, out_channels: int, in_channels: int = 32
) -> Model:

    if process == DENOISING:
        model = get_dip_model_denoising(out_channels, in_channels)
        return model
    elif process == INPAINTING:
        model = get_dip_model_inpainting(out_channels, in_channels)
        return model
    elif process == SR:
        model = get_dip_model_sr(out_channels, in_channels)
        return model
    
    raise TypeError(f'process should be one of {DENOISING}, {INPAINTING} or {SR}')


def dip(
    model: Model, 
    img_stem: str, 
    input_noise_torch: Tensor, 
    dtype,
    optimizer: optim.Optimizer,
    sigma: int = None, p: int = None, zoom: int = None, 
    downsampler: Optional[Model] = None,
    num_iter: int = 10_000, atleast: Optional[int] = None,
    exp_weight: float = 0.99, lr: float = 1e-2, reg_noise_std: float = 1/30,
    get_outputs_at: Optional[Union[int, List[int]]] = None,
    dir: Optional[Folder] = None
) -> Tuple[HtrDict, DataFrame, Any]:
    
    # decide the process according to the given parameters
    process = bu.get_process(sigma=sigma, p=p, zoom=zoom)
    
    # load the images
    img_true_np = bu.read_true_image(process, img_stem)
    img_true_np_psd_db_norm = fn.psd_db_norm(img_true_np)
    img_true_torch = imu.np_to_torch(img_true_np).type(dtype)
    img_true_in_channels = img_true_np.shape[0]
    
    img_noisy_np, tmp = bu.read_noisy_image(img_stem, sigma, p, zoom, ret_noise=True)
    img_noisy_torch = imu.np_to_torch(img_noisy_np).type(dtype)
    
    if process == INPAINTING:
        mask_np = tmp
        mask_torch = au.np_to_torch(mask_np).type(dtype)
        
    if process == SR:
        if downsampler is None:
            downsampler = su.get_downsampler(
                zoom=zoom, in_channels=img_true_in_channels
            ).type(dtype)
    
    # get the random output of the model
    random_output = mu.get_random_output(
        model=model,
        input_noise=input_noise_torch
    )
    if process == SR:
        tmp = imu.np_to_torch(random_output).type(dtype)
        tmp = downsampler(tmp)
        random_output = imu.torch_to_np(tmp)
        
    # get the metrics
    sim_metrics = metu.get_similarity_metrics()
    low_metrics = metu.get_lowpass_metrics()
    
    # we will store the metric results here
    metric_results: Dict[str, list] = defaultdict(list)
    # calculate the metrics
    for metric in sim_metrics + low_metrics:
        result = metu.calculate(metric, random_output, img_noisy_np)
        metric_results[metric].append(result)
    
    # train the model
    if process == DENOISING:
        htr: HtrDict = du.denoising(
            model=model, 
            optimizer=optimizer, 
            img_true_np=img_true_np,
            img_noisy_torch=img_noisy_torch,
            input_noise=input_noise_torch,
            num_iter=num_iter,
            atleast=atleast,
            exp_weight=exp_weight,
            reg_noise_std=reg_noise_std,
            get_outputs_at=get_outputs_at
        )
        
        metric_results['best psnr smooth'] = htr['best_psnr_gt_sm']
        metric_results['best psnr'] = htr['best_psnr_gt']
        metric_results['best iteration smooth'] = htr['best_iter_sm']
        metric_results['best iteration'] = htr['best_iter']
        metric_results = pd.DataFrame.from_dict(metric_results)
        
        grid = get_image_grid(
            [
                img_true_np, 
                htr['best_out'], 
                htr['best_out_sm'], 

                img_true_np_psd_db_norm, 
                fn.psd_db_norm(htr['best_out']), 
                fn.psd_db_norm(htr['best_out_sm'])
            ], 
            nrow=3
        )
    elif process == INPAINTING:
        htr = iu.inpainting(
            model=model,
            optimizer=optimizer,
            img_true_np=img_true_np,
            img_true_torch=img_true_torch,
            mask_torch=mask_torch,
            input_noise=input_noise_torch,
            num_iter=num_iter,
            atleast=atleast,
            exp_weight=exp_weight,
            reg_noise_std=reg_noise_std,
            get_outputs_at=get_outputs_at
        )
        
        metric_results['best psnr smooth'] = htr['best_psnr_gt_sm']
        metric_results['best psnr'] = htr['best_psnr_gt']
        metric_results['best iteration smooth'] = htr['best_iter_sm']
        metric_results['best iteration'] = htr['best_iter']
        metric_results = pd.DataFrame.from_dict(metric_results)
        
        grid = get_image_grid(
            [
                img_true_np, 
                htr['best_out'], 
                htr['best_out_sm'], 

                img_true_np_psd_db_norm, 
                fn.psd_db_norm(htr['best_out']), 
                fn.psd_db_norm(htr['best_out_sm'])
            ], 
            nrow=3
        )

    elif process == SR:
        htr = su.sr(
            model=model,
            optimizer=optimizer,
            img_true_np=img_true_np,
            img_noisy_torch=img_noisy_torch,
            input_noise=input_noise_torch,
            downsampler=downsampler,
            num_iter=num_iter,
            atleast=atleast,
            exp_weight=exp_weight,
            reg_noise_std=reg_noise_std,
            get_outputs_at=get_outputs_at
        )
        
        metric_results['best psnr smooth'] = htr['best_psnr_gt_sm']
        metric_results['best psnr'] = htr['best_psnr_gt']
        metric_results['best iteration smooth'] = htr['best_iter_sm']
        metric_results['best iteration'] = htr['best_iter']
        metric_results = pd.DataFrame.from_dict(metric_results)
        
        grid = get_image_grid(
            [
                img_true_np, 
                htr['best_out'], 
                htr['best_out_sm'], 

                img_true_np_psd_db_norm, 
                fn.psd_db_norm(htr['best_out']), 
                fn.psd_db_norm(htr['best_out_sm'])
            ], 
            nrow=3
        )
    
    if dir is not None:
        dir[HTR_PKL].save(htr)
        dir[METRICS_CSV].save(metric_results)
        dir[GRID_PNG].save(grid)
    
    return htr, metric_results, grid


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
    
    SAVE_OUT_EVERY = args.save_out_every
    if SAVE_OUT_EVERY != -1:
        SAVE_OUT_AT = list(range(1, NUM_ITER, SAVE_OUT_EVERY))
    else:
        SAVE_OUT_AT = args.save_out_at

        if len(SAVE_OUT_EVERY) > 0:
            SAVE_OUT_AT = list(map(int, SAVE_OUT_AT.split(',')))
        else:
            SAVE_OUT_AT = []

    REPEAT = args.repeat
    
    
    # decide the process according to the given parameters
    process = bu.get_process(sigma=SIGMA, p=P, zoom=ZOOM)
    
    # save the files here
    dir = ROOT[DIP][process]
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
            model = get_dip_model(process, out_channels=out_channels).type(DTYPE)
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

