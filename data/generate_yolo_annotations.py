import os
import pandas as pd
from PIL import Image

def convert_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height):
    """
    Converts bounding box coordinates to YOLO format.
    """
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return x_center, y_center, width, height

def process_excel_file(excel_path, image_dir):
    """
    Process an Excel file and generate YOLO annotation files.
    """
    data = pd.read_excel(excel_path)
    for _, row in data.iterrows():
        image_file = row['filename']
        class_id = int(row['class_id'])
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']

        image_path = os.path.join(image_dir, image_file)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        # Convert to YOLO format
        x_center, y_center, width, height = convert_to_yolo_format(
            xmin, ymin, xmax, ymax, img_width, img_height
        )

        # Write YOLO annotation file
        annotation_file = os.path.splitext(image_path)[0] + '.txt'
        with open(annotation_file, 'a') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def main():
    train_split_dir = '/Users/suhaibskhan/Documents/GitHub/Traffic-Sign-Detection-and-Recognition-System/data/GTSRB/Final_Training'

    for class_folder in os.listdir(train_split_dir):
        class_dir = os.path.join(train_split_dir, class_folder)
        if not os.path.isdir(class_dir):
            continue

        excel_files = [f for f in os.listdir(class_dir) if f.endswith('.xlsx') or f.endswith('.xls')]
        if not excel_files:
            print(f"No Excel file found in {class_dir}. Skipping.")
            continue

        excel_path = os.path.join(class_dir, excel_files[0])
        process_excel_file(excel_path, class_dir)

    print("YOLO annotations generated successfully.")

if __name__ == "__main__":
    main()