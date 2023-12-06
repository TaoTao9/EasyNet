# EasyNet: An Easy Network for 3D Industrial Anomaly Detection

Ruitao Chen* , Guoyang Xie* , Jiaqi Liu* , Jinbao Wang†, Ziqi Luo, Jinfan Wang, Feng Zheng† (* Equal contribution; † Corresponding authors)

Our paper has been accepted by ACM MM 2023 [[paper]](https://arxiv.org/abs/2307.13925).

## Datasets

**anomaly source dataset**
The Describable Textures dataset was used as the anomaly source image set in most of the experiments in the paper. You can run the follow code from the project directory to download the MVTec and the DTD datasets to the datasets folder in the project directory:

```shell
mkdir datasets
cd datasets
# Download describable textures dataset
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xf dtd-r1.0.1.tar.gz
rm dtd-r1.0.1.tar.gz
```


**MVTec 3D AD download**

- The `MVTec-3D AD` dataset can be download from the [Official Website of MVTec-3D AD](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad).

**eyecandies download**

- The `Eyecandies` dataset can be download from the [Official Website of Eyecandies](https://eyecan-ai.github.io/eyecandies/).

After download, put the dataset in `dataset` folder.

**dataset processing**

```shell
python utils/preprocessing.py datasets/mvtec3d/
```




## Pretrained models

[Links to model weights](https://drive.google.com/drive/folders/17N-4SgQDpjG0zf4uOVDDo0oNOeVQ1xN7?usp=drive_link)

## Evaluating

If you use the weights provided by us, please use Fusion1 mode for "--mode_type", and check the path where the "checkpoint.yaml" weights are configured.

During the test, please test according to the corresponding training mode, that is, "--mode_type" must be the same as during the training.

```shell
python test.py --gpu_id 0 --obj_id -1 --layer_size 2layer --mode_type Fusion1
python test.py --gpu_id 0 --obj_id -1 --layer_size 2layer --mode_type RGB

```

## Citations
Please consider citing our papers if you use the code:

```
@inproceedings{10.1145/3581783.3611876,
author = {Chen, Ruitao and Xie, Guoyang and Liu, Jiaqi and Wang, Jinbao and Luo, Ziqi and Wang, Jinfan and Zheng Feng},
title = {EasyNet: An Easy Network for 3D Industrial Anomaly Detection},
year = {2023},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {7038–7046},
numpages = {9},
series = {MM '23}
}
```

