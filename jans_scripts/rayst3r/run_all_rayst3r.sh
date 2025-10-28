#!/bin/bash
force=false
force_flag=""
if [ "$force" = true ]; then
    force_flag="--force"
fi

for scene_id_str in $(seq -f "%04g" 100 189); do
    python jans_scripts/rayst3r/zero_shot_rayst3r.py \
        --data_dir /data/graspnet/rayst3r_data \
        --output_dir /data/graspnet/output/GraspNet/rayst3r_zero_shot\
        --scene_id $scene_id_str \
        --view "0000" \
        ${force_flag}
done