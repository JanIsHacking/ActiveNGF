#!/bin/bash

mapping_depth_source="baseline"

for i in $(seq 100 189)
do
    scene_id=$(printf "%04d" $i)
    python -W ignore run.py configs/GraspNet/${mapping_depth_source}/scene_${scene_id}.yaml --force
done
