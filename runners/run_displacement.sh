#!/bin/bash
for i in {1..1}
do
    echo "At $i time for gradation FI7"
    blender --background --python scripts/main_displacement.py -- \
        --scene_scale 0.5 --global_scale 2 \
        --gradation FI7 --num_ballasts 600 --max_num_meshes 10 \
        --min_num_ballast_mats 2 --max_num_ballast_mats 4 \
        --image_width 2500 --image_height 2500 \
        --num_cam_images 2 --camera_height 0.7 \
        --render_labels --render_result --use_mixed_mats --mixed_mats_z_thresh_width 0.2 --disp_uv_scale 2 --label_thresh 0.2 \
        --use_hair_sys --hair_sys_density 10000 \
        --top_fine_num 100 --top_fine_min_d 0.1 --top_fine_max_d 0.375 --top_fine_disp_shift 0.2 #--save_scene --render_cam_traj --keep_largest_label --output_label_details --min_num_pixel_per_label 100

    echo "At $i time for gradation FI14"
    blender --background --python scripts/main_displacement.py -- \
        --scene_scale 0.5 --global_scale 2 \
        --gradation FI14 --num_ballasts 600 --max_num_meshes 10 \
        --min_num_ballast_mats 2 --max_num_ballast_mats 4 \
        --image_width 2500 --image_height 2500 \
        --num_cam_images 2 --camera_height 0.7 \
        --render_labels --render_result --use_mixed_mats --mixed_mats_z_thresh_width 0.2 --disp_uv_scale 2 --label_thresh 0.2 \
        --use_hair_sys --hair_sys_density 10000 \
        --top_fine_num 100 --top_fine_min_d 0.1 --top_fine_max_d 0.375 --top_fine_disp_shift 0.2 #--save_scene --render_cam_traj --keep_largest_label --output_label_details --min_num_pixel_per_label 100

    echo "At $i time for gradation FI23"
    blender --background --python scripts/main_displacement.py -- \
        --scene_scale 0.5 --global_scale 2 \
        --gradation FI23 --num_ballasts 500 --max_num_meshes 10 \
        --min_num_ballast_mats 1 --max_num_ballast_mats 3 \
        --image_width 2500 --image_height 2500 \
        --num_cam_images 2 --camera_height 0.7 \
        --render_labels --render_result --use_mixed_mats --mixed_mats_z_thresh_width 0.2 --disp_uv_scale 2 --label_thresh 0.2 \
        --use_hair_sys --hair_sys_density 10000 \
        --top_fine_num 200 --top_fine_min_d 0.1 --top_fine_max_d 0.375 --top_fine_disp_shift 0.2 #--save_scene --render_cam_traj --keep_largest_label --output_label_details --min_num_pixel_per_label 100

    echo "At $i time for gradation FI30"
    blender --background --python scripts/main_displacement.py -- \
        --scene_scale 0.5 --global_scale 2 \
        --gradation FI30 --num_ballasts 450 --max_num_meshes 10 \
        --min_num_ballast_mats 1 --max_num_ballast_mats 3 \
        --image_width 2500 --image_height 2500 \
        --num_cam_images 2 --camera_height 0.7 \
        --render_labels --render_result --use_mixed_mats --mixed_mats_z_thresh_width 0.2 --disp_uv_scale 2 --label_thresh 0.2 \
        --use_hair_sys --hair_sys_density 10000 \
        --top_fine_num 300 --top_fine_min_d 0.1 --top_fine_max_d 0.375 --top_fine_disp_shift 0.2 #--save_scene --render_cam_traj --keep_largest_label --output_label_details --min_num_pixel_per_label 100

    echo "At $i time for gradation FI39"
    blender --background --python scripts/main_displacement.py -- \
        --scene_scale 0.5 --global_scale 2 \
        --gradation FI39 --num_ballasts 450 --max_num_meshes 10 \
        --min_num_ballast_mats 1 --max_num_ballast_mats 3 \
        --image_width 2500 --image_height 2500 \
        --num_cam_images 2 --camera_height 0.7 \
        --render_labels --render_result --use_mixed_mats --mixed_mats_z_thresh_width 0.2 --disp_uv_scale 2 --label_thresh 0.2 \
        --use_hair_sys --hair_sys_density 10000 \
        --top_fine_num 400 --top_fine_min_d 0.1 --top_fine_max_d 0.375 --top_fine_disp_shift 0.2 #--save_scene --render_cam_traj --keep_largest_label --output_label_details --min_num_pixel_per_label 100
done 