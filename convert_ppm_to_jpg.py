import os
from PIL import Image

def convert_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".ppm"):
                input_path = os.path.join(root, file)
                # Preserve folder structure in output directory
                relative_path = os.path.relpath(root, input_dir)
                target_dir = os.path.join(output_dir, relative_path)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                output_path = os.path.join(target_dir, os.path.splitext(file)[0] + ".jpg")

                try:
                    with Image.open(input_path) as img:
                        img = img.convert("RGB")
                        img.save(output_path, "JPEG")
                        print(f"Converted {input_path} to {output_path}")
                except Exception as e:
                    print(f"Error converting {input_path}: {e}")

if __name__ == "__main__":
    input_directory = "/Users/suhaibskhan/Documents/GitHub/Traffic-Sign-Detection-and-Recognition-System/data/GTSRB_Split"
    output_directory = "/Users/suhaibskhan/Documents/GitHub/Traffic-Sign-Detection-and-Recognition-System/data/GTSRB_Split_JPG"

    convert_images(input_directory, output_directory)
    print("Conversion completed.")