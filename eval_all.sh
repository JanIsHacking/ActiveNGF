#!/bin/bash
export OMP_NUM_THREADS=8

for i in $(seq 100 189)
do
    scene_id=$(printf "%04d" $i)
    python eval.py \
        --scene_dir /workspace/output/GraspNet/scene_${scene_id}_nbv \
        --dataset_root /data/graspnet \
        --scene_id ${scene_id} \
        --config configs/GraspNet/scene_${scene_id}.yaml \
        --graspnet_checkpoint /workspace/ckpts2/checkpoint.tar
done