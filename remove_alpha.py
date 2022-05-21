from typing import Literal
from PIL import Image
from tqdm import tqdm

import utils.basic_utils as bu
from utils.paths import ROOT
from utils.keywords import *


def convert_rgb(img):
    return img.convert('RGB')


def convert_gray(img):
    return img.convert('L')

def convert(img, to=Literal['rgb', 'gray']):
    converter_map = {
        'rgb': convert_rgb,
        'gray': convert_gray,
    }
    
    return converter_map[to](img)

def main():
    for process in (DENOISING, INPAINTING, SR):
        print(f'{process = }')
        
        img_stems = bu.read_img_stems(process)
        
        for stem in tqdm(img_stems):
            name = bu.name_from_stem(stem)
            path = ROOT[IMAGES][process][TRUE][name].path
            
            tmp = stem.split('_')[-1]
            if tmp == 'rgb':
                to = 'rgb'
            else:
                to = 'gray'
            
            img = Image.open(path)
            img = convert(img, to=to)
            img.save(path)
            
if __name__ == '__main__':
    main()