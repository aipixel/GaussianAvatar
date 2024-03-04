<div align="center">

# <b>GaussianAvatar</b>: Towards Realistic Human Avatar Modeling from a Single Video via Animatable 3D Gaussians

[Liangxiao Hu](https://huliangxiao.github.io/)<sup>1</sup>, [Hongwen Zhang](https://zhanghongwen.cn/)<sup>2</sup>, [Yuxiang Zhang](https://zhangyux15.github.io/)<sup>3</sup>, [Boyao Zhou](https://morpheo.inrialpes.fr/people/zhou/)<sup>3</sup>, [Boning Liu](https://liuboning2.github.io/)<sup>3</sup>, [Shengping Zhang](http://homepage.hit.edu.cn/zhangshengping)<sup>1</sup>, [Liqiang Nie](https://liqiangnie.github.io/)<sup>1</sup>,

<sup>1</sup>Harbin Institute of Technology <sup>2</sup>Beijing Normal University <sup>3</sup>Tsinghua University

### [Projectpage](https://huliangxiao.github.io/GaussianAvatar) · [Paper](https://arxiv.org/abs/2312.02134) · [Video](https://www.youtube.com/watch?v=a4g8Z9nCF-k)

</div>

## :mega: Updates
[4/3/2024] The pretrained models for the other three people from People Snapshot are released on OneDrive.

[7/2/2024] The scripts for your own video are released.

[23/1/2024] Training and inference codes for People Snapshot are released.

## Introduction

We present GaussianAvatar, an efficient approach to creating realistic human avatars with dynamic 3D appearances from a single video. 

![](live_demo/gaussianavatar.gif)


## Installation

To deploy and run GaussianAvatar, run the following scripts:
```
conda env create --file environment.yml
conda activate gs-avatar
```

Then, compile ```diff-gaussian-rasterization``` and ```simple-knn``` as in [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) repository.

## Download models and data 

- SMPL/SMPL-X model: register and download [SMPL](https://smpl.is.tue.mpg.de/) and [SMPL-X](https://smpl-x.is.tue.mpg.de/), and put these files in ```assets/smpl_files```. The folder should have the following structure:
```
smpl_files
 └── smpl
   ├── SMPL_FEMALE.pkl
   ├── SMPL_MALE.pkl
   └── SMPL_NEUTRAL.pkl
 └── smplx
   ├── SMPLX_FEMALE.npz
   ├── SMPLX_MALE.npz
   └── SMPLX_NEUTRAL.npz
```

- Data: download the provided data from [OneDrive](https://hiteducn0-my.sharepoint.com/:f:/g/personal/lx_hu_hit_edu_cn/EsGcL5JGKhVGnaAtJ-rb1sQBR4MwkdJ9EWqJBIdd2mpi2w?e=KnloBM). These data include ```assets.zip```, ```gs_data.zip``` and ```pretrained_models.zip```. Please unzip ```assets.zip``` to the corresponding folder in the repository and unzip others to `gs_data_path` and `pretrained_models_path`.


## Run on People Snapshot dataset

We take the subject `m4c_processed` for example. 

### Training

```
python train.py -s $gs_data_path/m4c_processed -m output/m4c_processed --train_stage 1
```

### Evaluation

```
python eval.py -s $gs_data_path/m4c_processed -m output/m4c_processed --epoch 200
```

### Rendering novel pose

```
python render_novel_pose.py -s $gs_data_path/m4c_processed -m output/m4c_processed --epoch 200
```

## Run on Your Own Video

### Preprocessing

- masks and poses: use the bash script `scripts/custom/process-sequence.sh` in [InstantAvatar](https://github.com/tijiang13/InstantAvatar). The data folder should have the followings:
```
smpl_files
 ├── images
 ├── masks
 ├── cameras.npz
 └── poses_optimized.npz
```
- data format: we provide a script to convert the pose format of romp to ours (remember to change the `path` in L50 and L51):
```
cd scripts & python sample_romp2gsavatar.py
```
- position map of the canonical pose: (remember to change the corresponding `path`)
```
python gen_pose_map_cano_smpl.py
```

### Training for Stage 1

```
cd .. &  python train.py -s $path_to_data/$subject -m output/{$subject}_stage1 --train_stage 1 --pose_op_start_iter 10
```

### Training for Stage 2

- export predicted smpl:
```
cd scripts & python export_stage_1_smpl.py
```
- visualize the optimized smpl (optional):
```
python render_pred_smpl.py
```
- generate the predicted position map:
```
python gen_pose_map_our_smpl.py
```
- start to train
```
cd .. &  python train.py -s $path_to_data/$subject -m output/{$subject}_stage2 --train_stage 2 --stage1_out_path $path_to_stage1_net_save_path
```

## Todo

- [x] Release the reorganized code and data.
- [x] Provide the scripts for your own video.
- [ ] Provide the code for real-time annimation. 

## Citation

If you find this code useful for your research, please consider citing:
```
@article{hu2023gaussianavatar,
  title={GaussianAvatar: Towards Realistic Human Avatar Modeling from a Single Video via Animatable 3D Gaussians},
  author={Hu, Liangxiao and Zhang, Hongwen and Zhang, Yuxiang and Zhou, Boyao and Liu, Boning and Zhang, Shengping and Nie, Liqiang},
  journal={arXiv preprint arXiv:2312.02134},
  year={2023}
}
```

## Acknowledgements

This project is built on source codes shared by [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [POP](https://github.com/qianlim/POP), [HumanNeRF](https://github.com/chungyiweng/humannerf) and [InstantAvatar](https://github.com/tijiang13/InstantAvatar).
