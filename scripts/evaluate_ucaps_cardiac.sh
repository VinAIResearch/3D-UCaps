#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python evaluate.py --root_dir /home/ubuntu/Task02_Heart \
                    --gpus 1 \
                    --save_image 0 \
                    --model_name ucaps \
                    --dataset task02_heart \
                    --fold 0 \
                    --checkpoint_path /home/ubuntu/3D-UCaps/logs/ucaps_cardiac_fold_0.ckpt \
                    --val_patch_size 128 128 128 \
                    --sw_batch_size 2 \
                    --overlap 0.75

python evaluate.py --root_dir /home/ubuntu/Task02_Heart \
                    --gpus 1 \
                    --save_image 0 \
                    --model_name ucaps \
                    --dataset task02_heart \
                    --fold 1 \
                    --checkpoint_path /home/ubuntu/3D-UCaps/logs/ucaps_cardiac_fold_1.ckpt \
                    --val_patch_size 128 128 128 \
                    --sw_batch_size 2 \
                    --overlap 0.75

python evaluate.py --root_dir /home/ubuntu/Task02_Heart \
                    --gpus 1 \
                    --save_image 0 \
                    --model_name ucaps \
                    --dataset task02_heart \
                    --fold 2 \
                    --checkpoint_path /home/ubuntu/3D-UCaps/logs/ucaps_cardiac_fold_2.ckpt \
                    --val_patch_size 128 128 128 \
                    --sw_batch_size 2 \
                    --overlap 0.75

python evaluate.py --root_dir /home/ubuntu/Task02_Heart \
                    --gpus 1 \
                    --save_image 0 \
                    --model_name ucaps \
                    --dataset task02_heart \
                    --fold 3 \
                    --checkpoint_path /home/ubuntu/3D-UCaps/logs/ucaps_cardiac_fold_3.ckpt \
                    --val_patch_size 128 128 128 \
                    --sw_batch_size 2 \
                    --overlap 0.75