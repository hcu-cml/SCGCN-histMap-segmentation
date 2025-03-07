# SCGCN for Semantic Segmentation
## Introduction
This repository contains code of an SCGCN (Self-Constructing Graph Convolutional Network) called MSCG-Net (with ResNet-50 or ResNet-101 encoder backbone) for semantic segmentation of heterogeneous historical map corpora, and the pipeline of training and testing models. Please refer to our paper for details: 

[Semantic segmentation of historical maps using Self-Constructing Graph Convolutional Networks](https://doi.org/10.1080/15230406.2025.2468304)

## Code structure

```
├── config	      # config code
├── data		# dataset loader and pre-processing code
├── tools		# train and test code, ckpt and model_load
├── lib			# model block, loss, utils code, etc
└── ckpt 		# output check point, trained weights, log files, etc

```

## Environments

- python 3.5+
- pytorch 1.4.0
- opencv 3.4+
- tensorboardx 1.9
- albumentations 0.4.0
- pretrainedmodels 0.7.4
- others (see requirements.txt)

## Dataset preparation

Change DATASET_ROOT to your dataset path in ./data/PreProcessing/pre_process.py
```
DATASET_ROOT = '/your/path/to/historicalMaps'
```


## Train with a single GPU

```
CUDA_VISIBLE_DEVICES=0 python ./tools/train_R50.py  # trained weights test.pth
```

## Test with a single GPU

```
# To reproduce our results, download the trained-weights [test.pth](https://cloud.hcu-hamburg.de/nextcloud/s/Zet6zktajHzar52)
# and save them into ./ckpt folder before run test_submission.py
CUDA_VISIBLE_DEVICES=0 python ./tools/test_submission.py
```

## Citation: 
If you find the code helpful, please consider citing our work:

[Semantic segmentation of historical maps using Self-Constructing Graph Convolutional Networks](https://doi.org/10.1080/15230406.2025.2468304)
```
@article{arzoumanidis_2025_scgcn,
author = {Lukas Arzoumanidis and Julius Knechtel and Jan-Henrik Haunert and Youness Dehbi},
title = {Semantic Segmentation of Historical Maps using Self-Constructing Graph Convolutional Networks},
journal = {Cartography and Geographic Information Science},
year = {2025},
doi = {10.1080/15230406.2025.2468304},
}
```

[Self-Constructing Graph Convolutional Networks for Semantic Segmentation of Historical Maps](https://ica-abs.copernicus.org/articles/6/11/2023/)
```
@article{arzoumanidis_2023_scgcn,
author = {Lukas Arzoumanidis and Julius Knechtel and Jan-Henrik Haunert and Youness Dehbi},
title = {Self-Constructing Graph Convolutional Networks for Semantic Segmentation of Historical Maps},
journal = {Abstracts of the ICA. 31st International Cartographic Conference},
volume = {6},
year = {2023},
doi = {10.5194/ica-abs-6-11-2023}
}
```

## Other: 
The amazing [code](https://github.com/samleoqh/MSCG-Net) our project is based on and their papers:

[Multi-view Self-Constructing Graph Convolutional Networks with Adaptive Class Weighting Loss for Semantic Segmentation](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w5/Liu_Multi-View_Self-Constructing_Graph_Convolutional_Networks_With_Adaptive_Class_Weighting_Loss_CVPRW_2020_paper.pdf)


[Self-Constructing Graph Convolutional Networks for Semantic Labeling](https://arxiv.org/pdf/2003.06932)

