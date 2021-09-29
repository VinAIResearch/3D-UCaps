#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py --log_dir /home/ubuntu/logs \
                --gpus 8 \
                --accelerator ddp \
                --check_val_every_n_epoch 20 \
                --max_epochs 20000 \
                --dataset iseg2017 \
                --model_name ucaps \
                --root_dir /home/ubuntu/iseg \
                --cache_rate 1.0 \
                --train_patch_size 64 64 64 \
                --num_workers 64 \
                --batch_size 1 \
                --num_samples 8 \
                --in_channels 2 \
                --out_channels 4 \
                --val_patch_size 64 64 64 \
                --val_frequency 20 \
                --sw_batch_size 16 \
                --overlap 0.75