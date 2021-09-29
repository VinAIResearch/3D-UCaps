#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py --log_dir /home/ubuntu/logs \
                --gpus 8 \
                --accelerator ddp \
                --check_val_every_n_epoch 50 \
                --max_epochs 10000 \
                --dataset task02_heart \
                --model_name ucaps \
                --root_dir /home/ubuntu/Task02_Heart \
                --fold 0 \
                --cache_rate 1.0 \
                --train_patch_size 128 128 128 \
                --num_workers 64 \
                --batch_size 1 \
                --share_weight 0 \
                --num_samples 1 \
                --in_channels 1 \
                --out_channels 2 \
                --val_patch_size 128 128 128 \
                --val_frequency 50 \
                --sw_batch_size 2 \
                --overlap 0.75

python train.py --log_dir /home/ubuntu/logs \
                --gpus 8 \
                --accelerator ddp \
                --check_val_every_n_epoch 50 \
                --max_epochs 10000 \
                --dataset task02_heart \
                --model_name ucaps \
                --root_dir /home/ubuntu/Task02_Heart \
                --fold 1 \
                --cache_rate 1.0 \
                --train_patch_size 128 128 128 \
                --num_workers 64 \
                --batch_size 1 \
                --share_weight 0 \
                --num_samples 1 \
                --in_channels 1 \
                --out_channels 2 \
                --val_patch_size 128 128 128 \
                --val_frequency 50 \
                --sw_batch_size 2 \
                --overlap 0.75

python train.py --log_dir /home/ubuntu/logs \
                --gpus 8 \
                --accelerator ddp \
                --check_val_every_n_epoch 50 \
                --max_epochs 10000 \
                --dataset task02_heart \
                --model_name ucaps \
                --root_dir /home/ubuntu/Task02_Heart \
                --fold 2 \
                --cache_rate 1.0 \
                --train_patch_size 128 128 128 \
                --num_workers 64 \
                --batch_size 1 \
                --share_weight 0 \
                --num_samples 1 \
                --in_channels 1 \
                --out_channels 2 \
                --val_patch_size 128 128 128 \
                --val_frequency 50 \
                --sw_batch_size 2 \
                --overlap 0.75

python train.py --log_dir /home/ubuntu/logs \
                --gpus 8 \
                --accelerator ddp \
                --check_val_every_n_epoch 50 \
                --max_epochs 10000 \
                --dataset task02_heart \
                --model_name ucaps \
                --root_dir /home/ubuntu/Task02_Heart \
                --fold 3 \
                --cache_rate 1.0 \
                --train_patch_size 128 128 128 \
                --num_workers 64 \
                --batch_size 1 \
                --share_weight 0 \
                --num_samples 1 \
                --in_channels 1 \
                --out_channels 2 \
                --val_patch_size 128 128 128 \
                --val_frequency 50 \
                --sw_batch_size 2 \
                --overlap 0.75

