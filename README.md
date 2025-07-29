# Vision-Language Interactive Relation Mining for Open-Vocabulary Scene Graph Generation

## Installation and Setup

***Environment.***
This repo requires Pytorch>=1.9 and torchvision.

Then install the following packages:
```
pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo 
pip install transformers 
pip install SceneGraphParser spacy 
python setup.py build develop --user
```

***Pre-trained Visual-Semantic Space.*** Download the pre-trained `GLIP-T` and `GLIP-L` [checkpoints](https://github.com/microsoft/GLIP#model-zoo) into the ``MODEL`` folder. 
(!! GLIP has updated the downloading paths, please find these checkpoints following https://github.com/microsoft/GLIP#model-zoo)
```
mkdir MODEL
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth -O swin_tiny_patch4_window7_224.pth
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O swin_large_patch4_window12_384_22k.pth
```

## Dataset Preparation

1. Download original datasets
* ``Visual Genome (VG)``: Download the original [VG](https://visualgenome.org/) data into ``DATASET/VG150`` folder. Refer to [vg_prepare](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/DATASET.md).

The `DATASET` directory is organized roughly as follows:
```
─VG150
    ├─VG_100K
    ├─weak_supervisions
    ├─image_data.json
    ├─VG-SGG-dicts-with-attri.json
    ├─region_descriptions.json
    └─VG-SGG-with-attri.h5 
```

Since GLIP pre-training has seen part of VG150 test images, we remove these images and get new VG150 split and write it to `VG-SGG-with-attri.h5`. 
Please refer to [tools/cleaned_split_GLIPunseen.ipynb](tools/cleaned_split_GLIPunseen.ipynb).


## Training & Evaluation

1.Open-vocabulary SGG

```
# VL-IRM$ (Swin-T) 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port 10038 tools/train_net.py     --task_config configs/vg150/finetune.yaml --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml     SOLVER.IMS_PER_BATCH 1 TEST.IMS_PER_BATCH 1     MODEL.DYHEAD.RELATION_REP_REFINER False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False OUTPUT_DIR OUTPUT/OIV6_lr5e5   SOLVER.CHECKPOINT_PERIOD 5000 MODEL.DYHEAD.SGG_MODE 'sgdet' SOLVER.ov_relation True SOLVER.usetrain True TEST.myeval False OUTPUT_log output_test SOLVER.more_base_rel False
```
## Acknowledgement


uploading...

This repo is based on [GLIP](https://github.com/microsoft/GLIP), [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), [SGG_from_NLS](https://github.com/YiwuZhong/SGG_from_NLS), [VS3](https://github.com/zyong812/VS3_CVPR23). Thanks for their contribution.
