#!/bin/bash

rotate_angle=$1
axis="$2"

export CUDA_VISIBLE_DEVICES=0

python evaluate_iseg.py --root_dir /home/ubuntu/iseg \
                    --gpus 1 \
                    --save_image 0 \
                    --model_name ucaps \
                    --checkpoint_path /home/ubuntu/3D-UCaps/logs/ucaps_iseg_128_128_128.ckpt \
                    --val_patch_size 128 128 128 \
                    --sw_batch_size 2 \
                    --overlap 0.25 \
                    --rotate_angle $rotate_angle \
                    --axis $axis

python evaluate_iseg.py --root_dir /home/ubuntu/iseg \
                    --gpus 1 \
                    --save_image 0 \
                    --model_name unet \
                    --checkpoint_path /home/ubuntu/3D-UCaps/logs/unet_iseg_128_128_128.ckpt \
                    --val_patch_size 128 128 128 \
                    --sw_batch_size 8 \
                    --overlap 0.25 \
                    --rotate_angle $rotate_angle \
                    --axis $axis