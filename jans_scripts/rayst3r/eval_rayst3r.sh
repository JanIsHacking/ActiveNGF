#!/bin/bash
export OMP_NUM_THREADS=8
force=true
force_flag=""
if [ "$force" = true ]; then
    force_flag="--force"
fi

# read the scene id from the command line
scene_id=$1
if [ -z "$scene_id" ]; then
    echo "Scene id is required"
    return 1 2> /dev/null || exit 1
fi

if [ "$scene_id" -lt 100 ] || [ "$scene_id" -gt 189 ]; then
    echo "Scene id must be between 100 and 189"
    return 1 2> /dev/null || exit 1
fi

scene_id_str=$(printf "%04d" $scene_id)
python eval_rayst3r.py \
    --data_dir /data/graspnet/rayst3r_data \
    --dataset_root /data/graspnet \
    --scene_id ${scene_id_str} \
    --graspnet_checkpoint /workspace/ckpts2/checkpoint.tar \
    ${force_flag}