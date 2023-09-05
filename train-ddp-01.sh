#!/bin/bash
source /opt/conda/bin/activate
cd /workspace/mnt/storage/shihao/MyCode-01/FishDreamer
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install pytorch-lightning
pip install albumentations==0.5.2 --no-binary qudida,albumentations
pip install -U scikit-learn==0.24.2
pip install webdataset
pip install easydict
pip install pandas
pip install kornia==0.5.0
pip install timm
pip install -U openmim
mim install mmcv-full
pip install hydra-core==1.1.0
pip install einops
export USER=$(whoami)
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
python3 bin/train.py -cn FishDreamer-KITTI-SQ-lr10 data.batch_size=3 trainer.kwargs.max_epochs=150 data.train.transform_variant=resize