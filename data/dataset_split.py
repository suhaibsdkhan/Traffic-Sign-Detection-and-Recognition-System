# classification/dataset_split.py

import os
import shutil
import random

def split_gtsrb(data_root, output_root, train_ratio=0.8):
    if not os.path.exists(data_root):
        print(f"Error: The data_root path '{data_root}' does not exist.")
        return

    classes = os.listdir(data_root)
    if not classes:
        print(f"Error: No class folders found in '{data_root}'.")
        return

    print(f"Found {len(classes)} classes.")

    for cls in classes:
        cls_path = os.path.join(data_root, cls)
        if not os.path.isdir(cls_path):
            print(f"Skipping '{cls_path}' as it is not a directory.")
            continue

        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.ppm','.jpg','.png'))]
        if not images:
            print(f"No images found in '{cls_path}'. Skipping this class.")
            continue

        print(f"Class '{cls}': {len(images)} images found.")

        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        print(f"Splitting '{cls}': {len(train_imgs)} train, {len(val_imgs)} val.")

        train_cls_dir = os.path.join(output_root, 'train', cls)
        val_cls_dir = os.path.join(output_root, 'val', cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)

        # Copy training images
        for img in train_imgs:
            src_path = os.path.join(cls_path, img)
            dst_path = os.path.join(train_cls_dir, img)
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Failed to copy '{src_path}' to '{dst_path}': {e}")

        # Copy validation images
        for img in val_imgs:
            src_path = os.path.join(cls_path, img)
            dst_path = os.path.join(val_cls_dir, img)
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Failed to copy '{src_path}' to '{dst_path}': {e}")

        print(f"Completed splitting for class '{cls}'.\n")

    print("Dataset splitting completed successfully.")

if __name__ == '__main__':
    # ======================
    # User Configuration
    # ======================

    # Replace these paths with your actual paths
    data_root = '/Users/suhaibskhan/Documents/GitHub/Traffic-Sign-Detection-and-Recognition-System/data/GTSRB/Final_Training/Images'  # <-- Replace with your path
    output_root = '/Users/suhaibskhan/Documents/GitHub/Traffic-Sign-Detection-and-Recognition-System/data/GTSRB_Split'               # <-- Replace with your path

    # Optional: Set a seed for reproducibility
    random.seed(42)

    # Run the split
    split_gtsrb(data_root, output_root, train_ratio=0.8)