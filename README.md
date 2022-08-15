# ISNAS-DIP: Image-Specific Neural Architecture Search for Deep Image Prior (CVPR 2022)  

#### Metin Ersin Arican*, Ozgur Kara*, Gustav Bredell, Ender Konukoglu  

#### [\[Paper\]](https://arxiv.org/abs/2111.15362v2) [\[Dataset\]](https://zenodo.org/record/6351599#.YkG1xN9Bzct) [\[Presentation\]](https://www.youtube.com/watch?v=vuTeMda-cVo) 

---

## News

This repo is the official implementation of ISNAS-DIP. 

---

### Overview

![Visualization of proposed metrics](https://github.com/ozgurkara99/ISNAS-DIP/blob/main/docs/method.jpg)

Recent works show that convolutional neural network (CNN) architectures have a spectral bias towards lower frequencies, which has been leveraged for various image restoration tasks in the Deep Image Prior (DIP) framework. The benefit of the inductive bias the network imposes in the DIP framework depends on the architecture. Therefore, researchers have studied how to automate the search to determine the best-performing model. However, common neural architecture search (NAS) techniques are resource and time-intensive. Moreover, best-performing models are determined for a whole dataset of images instead of for each image independently, which would be prohibitively expensive. In this work, we first show that optimal neural architectures in the DIP framework are image-dependent. Leveraging this insight, we then propose an image-specific NAS strategy for the DIP framework that requires substantially less training than typical NAS approaches, effectively enabling image-specific NAS. We justify the proposed strategy's effectiveness by (1) demonstrating its performance on a NAS Dataset for DIP that includes 522 models from a particular search space (2) conducting extensive experiments on image denoising, inpainting, and super-resolution tasks. Our experiments show that image-specific metrics can reduce the search space to a small cohort of models, of which the best model outperforms current NAS approaches for image restoration. 

## Getting Started

### Installation
1- Clone the repo:  
```
git clone https://github.com/ozgurkara99/ISNAS-DIP.git
```  
2- Create a conda (suggested) environment and install the required packages:  
```
conda create -n isnasdip python=3.8
pip install -r requirements.txt
```  
3- If any of the packages listed in requirements.txt is failed to installed, install it manually, remove it from the txt file and rerun the above command.  
4- Go to utils/paths.py and change the variable PROJECT_FOLDER to path of the current directory.  

### Usage
- To run isnasdip experiment see the isnasdip.sh  
- To run nasdip experiment see the nasdip.sh  
- To run dip experiment see the dip.sh  

### Citation:
If you use our [paper](https://arxiv.org/abs/2111.15362) or [dataset](https://zenodo.org/record/6351599#.YkG1xN9Bzct), please consider citing our paper: 
```
@inproceedings{arican2022isnasdip,
  title={ISNAS-DIP: Image-Specific Neural Architecture Search for Deep Image Prior},
  author={Arican, Metin and Kara, Ozgur and Bredell, Gustav and Konukoglu, Ender},
  booktitle= {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Acknowledgements  

nasdip.py and dip.py scripts borrow some codes from [Chen et. al](https://github.com/YunChunChen/NAS-DIP-pytorch) and [Ulyanov et. al](https://github.com/DmitryUlyanov/deep-image-prior).
