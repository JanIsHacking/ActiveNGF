#!/bin/bash

for i in $(seq 101 189)
do
    scene_id=$(printf "%04d" $i)
    python -W ignore run.py configs/GraspNet/scene_${scene_id}.yaml
done
