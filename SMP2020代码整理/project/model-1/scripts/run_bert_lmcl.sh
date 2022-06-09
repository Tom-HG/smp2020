#!/bin/bash
# encoding: utf-8

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
PROJECT_FOLDER=$(dirname ${SHELL_FOLDER})
cd ${PROJECT_FOLDER}

model_save_path=model_save/models_roberta_xlarge_lmcl
mkdir -p ${model_save_path}
# 以下自行修改
export PATH=/home/arlencai/miniconda3/bin:$PATH

python train_lmcl.py \
       --bert_path=pretrain/pytorch_chinese_L-24_H-1024_A-16 \
       --batch_size=8 \
       --model_save_path=${model_save_path} \
       --gpu_devices=0,1,2,3 \
       --do_train

# --bert_path=/apdcephfs/share_470749/rossliang/pretrain/chinese_wwm_pytorch
