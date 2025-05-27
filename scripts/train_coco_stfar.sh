# #!/bin/bash        
corruption_names=("gaussian_noise" "shot_noise" "impulse_noise" "defocus_blur" 
                  "glass_blur" "motion_blur" "zoom_blur" "snow" 
                  "frost" "fog" "brightness" "contrast" "elastic_transform" 
                  "pixelate" "jpeg_compression")

counter=0

for corrupt in "${corruption_names[@]}"; do
    echo "Testing with corrupt option: $corrupt"

    CUDA_VISIBLE_DEVICES=0 python tools/train.py --config ./configs/tta/res50_coco_stfar.py --work-dir work_dirs/coco/stfar/${counter}_${corrupt} --cfg-options corrupt=${corrupt}

    counter=$((counter + 1))
done


