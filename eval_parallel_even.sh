#!/bin/bash
export OMP_NUM_THREADS=8  # use 8 threads per process

max_jobs=4

for scene_id_str in $(seq -f "%04g" 101 189); do
    # convert to decimal (strip leading zeros)
    scene_id=$((10#$scene_id_str))

    # process only even scene IDs
    if (( scene_id % 2 == 0 )); then
        python eval.py \
            --scene_dir /workspace/output/GraspNet/scene_${scene_id_str}_nbv \
            --dataset_root /data/graspnet \
            --scene_id ${scene_id_str} \
            --config configs/GraspNet/scene_${scene_id_str}.yaml \
            --graspnet_checkpoint /workspace/ckpts2/checkpoint.tar \
            --force &
    fi

    # limit to $max_jobs parallel processes
    while (( $(jobs -r | wc -l) >= max_jobs )); do
        sleep 1
    done
done

wait  # wait for all background jobs to finish
