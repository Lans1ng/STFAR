#!/bin/bash

corruption_names=("gaussian_noise" "shot_noise" "impulse_noise" "defocus_blur" 
                  "glass_blur" "motion_blur" "zoom_blur" "snow" 
                  "frost" "fog" "brightness" "contrast" "elastic_transform" 
                  "pixelate" "jpeg_compression")

counter=0

prev_ckpt=""

for corrupt in "${corruption_names[@]}"; do
    echo "Testing with corruption: $corrupt"

    work_dir="work_dirs/coco/stfar_ctta/${counter}_${corrupt}"

    if [ $counter -eq 0 ]; then
        # First run: just load pretrained model
        CUDA_VISIBLE_DEVICES=0 python tools/train.py \
            --config ./configs/tta/res50_coco_stfar_ctta.py \
            --work-dir ${work_dir} \
            --cfg-options corrupt=${corrupt}
    else
        # Subsequent runs: load the previous checkpoint and continue training
        CUDA_VISIBLE_DEVICES=0 python tools/train.py \
            --config ./configs/tta/res50_coco_stfar_ctta.py \
            --work-dir ${work_dir} \
            --cfg-options corrupt=${corrupt} load_from=${prev_ckpt}
    fi

    # Update prev_ckpt assuming the last checkpoint is saved as latest.pth
    prev_ckpt="${work_dir}/latest.pth"

    counter=$((counter + 1))
done
