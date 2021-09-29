#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py --log_dir /home/ubuntu/logs \
                --gpus 1 \
                --check_val_every_n_epoch 1 \
                --max_epochs 200 \
                --dataset luna16 \
                --model_name ucaps \
                --root_dir /home/ubuntu/LUNA16 \
                --fold 0 \
                --cache_dir /home/ubuntu/cache_dir \
                --train_patch_size 128 128 128 \
                --num_workers 4 \
                --batch_size 1 \
                --share_weight 0 \
                --num_samples 1 \
                --in_channels 1 \
                --out_channels 2 \
                --val_patch_size 128 128 128 \
                --val_frequency 1 \
                --sw_batch_size 2 \
                --overlap 0.75