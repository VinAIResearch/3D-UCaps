#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python evaluate.py --root_dir /home/ubuntu/iseg \
                    --gpus 1 \
                    --save_image 0 \
                    --model_name ucaps \
                    --dataset iseg2017 \
                    --checkpoint_path /home/ubuntu/3D-UCaps/logs/ucaps_iseg_64_64_64.ckpt \
                    --val_patch_size 64 64 64 \
                    --sw_batch_size 16 \
                    --overlap 0.75