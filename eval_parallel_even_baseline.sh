#!/bin/bash
export OMP_NUM_THREADS=8  # use 8 threads per process

max_jobs=4
mapping_depth_source="baseline"
checkpoint_path="/workspace/ckpts2/checkpoint.tar"
force=false
force_flag=""
if [ "$force" = true ]; then
    force_flag="--force"
fi

for scene_id_str in $(seq -f "%04g" 100 189); do
    # convert to decimal (strip leading zeros)
    scene_id=$((10#$scene_id_str))

    # process only even scene IDs
    if (( scene_id % 2 == 0 )); then
        python eval.py \
            --scene_dir /workspace/output/GraspNet/${mapping_depth_source}/scene_${scene_id_str}_nbv \
            --dataset_root /data/graspnet \
            --scene_id ${scene_id_str} \
            --config configs/GraspNet/${mapping_depth_source}/scene_${scene_id_str}.yaml \
            --graspnet_checkpoint ${checkpoint_path} \
            ${force_flag} &
    fi

    # limit to $max_jobs parallel processes
    while (( $(jobs -r | wc -l) >= max_jobs )); do
        sleep 1
    done
done

wait  # wait for all background jobs to finish
