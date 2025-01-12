# classification/generate_image_lists.py

import os

def generate_image_list(split_dir, output_file):
    with open(output_file, 'w') as f:
        for cls in os.listdir(split_dir):
            cls_path = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            for img in os.listdir(cls_path):
                if img.lower().endswith(('.ppm', '.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(cls_path, img)
                    f.write(f"{os.path.abspath(img_path)}\n")

def main():
    train_split_dir = '/Users/suhaibskhan/Documents/GitHub/Traffic-Sign-Detection-and-Recognition-System/data/GTSRB_Split_JPG/train'
    val_split_dir = '/Users/suhaibskhan/Documents/GitHub/Traffic-Sign-Detection-and-Recognition-System/data/GTSRB_Split_JPG/val'
    train_output = '/Users/suhaibskhan/Documents/GitHub/Traffic-Sign-Detection-and-Recognition-System/data/train.txt'
    val_output = '/Users/suhaibskhan/Documents/GitHub/Traffic-Sign-Detection-and-Recognition-System/data/val.txt'
    
    generate_image_list(train_split_dir, train_output)
    generate_image_list(val_split_dir, val_output)
    print("train.txt and val.txt generated successfully.")

if __name__ == '__main__':
    main()