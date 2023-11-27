# EasyNet: An Easy Network for 3D Industrial Anomaly Detection

Ruitao Chen* , Guoyang Xie* , Jiaqi Liu* , Jinbao Wang†, Ziqi Luo, Jinfan Wang, Feng Zheng† (* Equal contribution; † Corresponding authors)

Our paper has been accepted by ACM MM 2023 [[paper]](https://arxiv.org/abs/2307.13925).

## Datasets

**MVTec 3D AD download**

**eyecandies download**

**dataset processing**

## Training

### **train RGB branch**

```
python trainer_rgb_fu.py --gpu_id (your gpu id) --obj_id (dataset class id) --layer_size 2layer --mode_type RGB
```



### **train depth branch**

```
python trainer_rgb_fu.py --gpu_id (your gpu id) --obj_id (dataset class id) --layer_size 2layer --mode_type Depth
```



### **train fusion branch**

train type1：

```
python trainer_rgb_fu.py --gpu_id (your gpu id) --obj_id (dataset class id) --layer_size 2layer --mode_type RGBD
```

train type2：

```
python trainer_rgb_fu.py --gpu_id (your gpu id) --obj_id (dataset class id) --layer_size 2layer --mode_type RGBD
```



## Pretrained models

## Evaluating

```
python test.py --bs (dataset class id)\\
--gpu_id (your gpu id)\\

```



```
@inproceedings{10.1145/3581783.3611876,
author = {Chen, Ruitao and Xie, Guoyang and Liu, Jiaqi and Wang, Jinbao and Luo, Ziqi and Wang, Jinfan and Zheng, Feng},
title = {EasyNet: An Easy Network for 3D Industrial Anomaly Detection},
year = {2023},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {7038–7046},
numpages = {9},
series = {MM '23}
}
```

