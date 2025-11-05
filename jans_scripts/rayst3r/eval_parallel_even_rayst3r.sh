#!/bin/bash
export OMP_NUM_THREADS=8  # use 8 threads per process
export CUDA_VISIBLE_DEVICES=0

max_jobs=4
force=true

checkpoint_path="/workspace/ckpts2/checkpoint.tar"
predictions_dir="/data/graspnet/output/GraspNet/rayst3r_zero_shot"
dataset_root="/data/graspnet"

force_flag=""
if [ "$force" = true ]; then
    force_flag="--force"
fi

for scene_id_str in $(seq -f "%04g" 100 189); do
    # convert to decimal (strip leading zeros)
    scene_id=$((10#$scene_id_str))

    # process only even scene IDs
    if (( scene_id % 2 == 0 )); then
        python jans_scripts/rayst3r/eval_rayst3r.py \
            --predictions_dir ${predictions_dir} \
            --dataset_root ${dataset_root} \
            --scene_id ${scene_id_str} \
            --graspnet_checkpoint ${checkpoint_path} \
            ${force_flag} &
    fi

    # limit to $max_jobs parallel processes
    while (( $(jobs -r | wc -l) >= max_jobs )); do
        sleep 1
    done
done

wait  # wait for all background jobs to finish
