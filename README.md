# Gaussian Grouping [ECCV'24]

> [**Gaussian Grouping: Segment and Edit Anything in 3D Scenes**](https://arxiv.org/abs/2312.00732)           
> [[Project Page]](https://ymq2017.github.io/gaussian-grouping)           
> ECCV 2024  
> ETH Zurich

## Installation

Clone the repository locally
```
git clone https://github.com/lkeab/gaussian-grouping.git
cd gaussian-grouping
```

Our default, provided install method is based on Conda package and environment management:
```bash
conda create -n gaussian_grouping python=3.8 -y
conda activate gaussian_grouping 

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install plyfile==0.8.1
pip install tqdm scipy wandb opencv-python scikit-learn lpips

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

(Optional) If you want to prepare masks on your own dataset, you will also need to prepare [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) environment.

```bash
cd Tracking-Anything-with-DEVA
pip install -e .
bash scripts/download_models.sh     # Download the pretrained models

git clone https://github.com/hkchengrex/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO

cd ../..
```

(Optional) If you want to inpaint on your own dataset, you will also need to prepare [LaMa](https://github.com/advimman/lama) environment.

```bash
cd lama
pip install -r requirements.txt
cd ..
```

## Training

### 1. Prepare associated SAM masks

#### 1.1 Pre-converted datasets
We provide converted datasets in our paper, You can use directly train on datasets from [hugging face link](https://huggingface.co/mqye/Gaussian-Grouping/tree/main)

```
data
|____bear
|____lerf
| |____figurines
|____mipnerf360
| |____counter
```


#### 1.2 (Optional) Prepare your own datasets
For your custom dataset, you can follow this step to create masks for training. If you want to prepare masks on your own dataset, you will need [DEVA](../Tracking-Anything-with-DEVA/README.md) python environment and checkpoints.


```
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```
Firstly, convert initial camera pose and point cloud with colmap
```
python convert.py -s <location>
```

Then, convert SAM associated object masks. Note that the quality of converted-mask will largely affect the results of 3D segmentation and editing. And getting the mask is very fast. So it is best to adjust the parameters of anything segment first to make the mask as consistent and reasonable as possible from multiple views.

Example1. Bear dataset
```
bash script/prepare_pseudo_label.sh bear 1
```

Example2. figurines dataset
```
bash script/prepare_pseudo_label.sh lerf/figurines 1
```

Example3. counter dataset
```
bash script/prepare_pseudo_label.sh mipnerf360/counter 2
```

### 2. Training and Masks Rendering

For Gaussian Grouping training and segmentation rendering of trained 3D Gaussian Grouping model:

Example1. Bear dataset
```
bash script/train.sh bear 1
```

Example2. figurines dataset
```
bash script/train_lerf.sh lerf/figurines 1
```

Example3. counter dataset
```
bash script/train.sh mipnerf360/counter 2
```

