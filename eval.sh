#!/bin/bash
export OMP_NUM_THREADS=8
force=true
force_flag=""
if [ "$force" = true ]; then
    force_flag="--force"
fi
mapping_depth_source="gt"

python eval.py \
    --scene_dir /workspace/output/GraspNet/${mapping_depth_source}/scene_0100_nbv \
    --dataset_root /data/graspnet \
    --scene_id 0100 \
    --config configs/GraspNet/${mapping_depth_source}/scene_0100.yaml \
    --graspnet_checkpoint /workspace/ckpts2/checkpoint.tar \
    ${force_flag}