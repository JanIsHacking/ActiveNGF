#!/bin/bash

mapping_depth_source="baseline"
random_nbv=true
force=true
force_flag=""
if [ "$force" = true ]; then
    force_flag="--force"
fi

random_nbv_str=""
if [ "$random_nbv" = true ]; then
    random_nbv_str="_random_nbv"
fi
mapping_depth_source_str="${mapping_depth_source}${random_nbv_str}"

for i in $(seq 100 189)
do
    scene_id=$(printf "%04d" $i)
    python -W ignore run.py configs/GraspNet/${mapping_depth_source_str}/scene_${scene_id}.yaml ${force_flag}
done
