#!/bin/bash

lr=0.2
Lambda=0.25
cld_t=0.4
clusters=50
CUDA_VISIBLE_DEVICES=0,1 \
python main_imagenet_moco_cld.py \
  -a resnet50 \
  --lr ${lr} \
  --workers 8 \
  --batch-size 128 \
  --moco-k 65536 \
  --Lambda 0.25 \
  --aug-plus --cos --mlp \
  --moco-t 0.2 \
  --cld-t ${cld_t} \
  --amp-opt-level O1 \
  --num-iters 5 \
  --clusters ${clusters} \
  --use-kmeans \
  --normlinear \
  --save-dir ~/ddsm_CLD/trial_1/c_50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --data ../DDSM_patches

