#!/bin/bash
export OMP_NUM_THREADS=8
python eval.py \
    --scene_dir /workspace/output/GraspNet/scene_0100_nbv \
    --dataset_root /data/graspnet \
    --scene_id 0100 \
    --config configs/GraspNet/scene_0100.yaml