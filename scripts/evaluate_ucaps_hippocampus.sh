#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python evaluate.py --root_dir /home/ubuntu/Task04_Hippocampus \
                    --gpus 1 \
                    --save_image 0 \
                    --model_name ucaps \
                    --dataset task04_hippocampus \
                    --fold 0 \
                    --checkpoint_path /home/ubuntu/3D-UCaps/logs/ucaps_hippocampus_fold_0.ckpt \
                    --val_patch_size 32 32 32 \
                    --sw_batch_size 8 \
                    --overlap 0.75

python evaluate.py --root_dir /home/ubuntu/Task04_Hippocampus \
                    --gpus 1 \
                    --save_image 0 \
                    --model_name ucaps \
                    --dataset task04_hippocampus \
                    --fold 1 \
                    --checkpoint_path /home/ubuntu/3D-UCaps/logs/ucaps_hippocampus_fold_1.ckpt \
                    --val_patch_size 32 32 32 \
                    --sw_batch_size 8 \
                    --overlap 0.75

python evaluate.py --root_dir /home/ubuntu/Task04_Hippocampus \
                    --gpus 1 \
                    --save_image 0 \
                    --model_name ucaps \
                    --dataset task04_hippocampus \
                    --fold 2 \
                    --checkpoint_path /home/ubuntu/3D-UCaps/logs/ucaps_hippocampus_fold_2.ckpt \
                    --val_patch_size 32 32 32 \
                    --sw_batch_size 8 \
                    --overlap 0.75

python evaluate.py --root_dir /home/ubuntu/Task04_Hippocampus \
                    --gpus 1 \
                    --save_image 0 \
                    --model_name ucaps \
                    --dataset task04_hippocampus \
                    --fold 3 \
                    --checkpoint_path /home/ubuntu/3D-UCaps/logs/ucaps_hippocampus_fold_3.ckpt \
                    --val_patch_size 32 32 32 \
                    --sw_batch_size 8 \
                    --overlap 0.75