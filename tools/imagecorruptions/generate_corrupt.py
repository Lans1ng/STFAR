from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
import numpy as np
import os
from tqdm import tqdm  

#coco
# image_dir = "../../../_DATASET/COCO_C_5/images/val2017"
# corrupt_dir = "../../../_DATASET/COCO_C_5/images"

#voc
image_dir = "../../../_DATASET/VOCdevkit/VOC2007/JPEGImages"
corrupt_dir = "../../../_DATASET/VOCdevkit/VOC2007/JPEGImages-C"

image_list = os.listdir(image_dir)
# print(image_list)

# Ensure that the corrupt directory exists
os.makedirs(corrupt_dir, exist_ok=True)

# Get corruption names
corruptions = get_corruption_names('common')

# Outer loop for corruption types with progress bar
for corruption in tqdm(corruptions, desc="Corruption Types"):
    # Create a directory for each corruption type
    corruption_dir = os.path.join(corrupt_dir, corruption)
    os.makedirs(corruption_dir, exist_ok=True)
    
    # Inner loop for processing images with progress bar
    for image_path in tqdm(image_list, desc=f"Processing {corruption}", leave=False):
        if image_path.lower().endswith('.jpg'):
            # Full path to the image
            full_image_path = os.path.join(image_dir, image_path)
            
            # Load and corrupt the image
            image = np.asarray(Image.open(full_image_path))
            corrupted = corrupt(image, corruption_name=corruption, severity=5)
            
            # Convert back to an image
            corrupted_image = Image.fromarray(corrupted)
            
            # Save the corrupted image in the corresponding corruption folder
            corrupted_image_path = os.path.join(corruption_dir, image_path)
            corrupted_image.save(corrupted_image_path)
        else:
            continue
