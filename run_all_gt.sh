#!/bin/bash

for i in $(seq 100 189)
do
    scene_id=$(printf "%04d" $i)
    python -W ignore run.py configs/GraspNet/gt/scene_${scene_id}.yaml
done
