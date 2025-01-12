# **Traffic Sign Detection and Recognition System**

This project implements a deep learning-based system for detecting and recognizing traffic signs. It combines YOLOv5 for object detection and a ResNet classifier for traffic sign classification. The dataset used is the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

---

## **Table of Contents**

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Usage](#usage)
   - [Convert `.ppm` to `.jpg`](#convert-ppm-to-jpg)
   - [Generate YOLO Annotations](#generate-yolo-annotations)
   - [Train YOLOv5](#train-yolov5)
   - [Train ResNet Classifier](#train-resnet-classifier)
   - [Real-Time Detection and Classification](#real-time-detection-and-classification)
6. [To-Do](#to-do)
7. [License](#license)
8. [Credits](#credits)

---

## **Features**

- Object detection using **YOLOv5**.
- Traffic sign classification using **ResNet-18**.
- Real-time inference on video streams or webcam.
- GUI interface for uploading images and visualizing results.
- Dataset preprocessing, including Excel-based annotation parsing and image format conversion.

---

## **Project Structure**

```
Traffic-Sign-Detection-and-Recognition-System/
├── data/                      # Dataset-related scripts and configurations
│   ├── data_config.yaml       # YOLOv5 dataset config
│   ├── convert_ppm_to_jpg.py  # Converts .ppm images to .jpg
│   ├── generate_annotations_from_excel.py  # Generates YOLO annotations from Excel files
│   ├── generate_image_lists.py  # Creates train.txt and val.txt
│   └── README.md
├── classification/            # ResNet classifier scripts
│   ├── train_classifier.py
│   ├── test_classifier.py
│   ├── models/                # Stores trained classifier weights
│   └── README.md
├── inference/                 # Scripts for real-time inference and GUI
│   ├── real_time_inference.py
│   ├── gui_app.py
│   └── README.md
├── yolov5/                    # YOLOv5 repository (if cloned locally)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files and folders to ignore in Git
└── README.md                  # Project description (this file)
```

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Traffic-Sign-Detection-and-Recognition-System.git
   cd Traffic-Sign-Detection-and-Recognition-System
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## **Dataset Preparation**

1. **Download the GTSRB Dataset**:  
   [GTSRB Dataset Download Link](https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)

2. **Organize the Dataset**:  
   Extract the dataset and place it under the `data/` folder as follows:
   ```
   data/
   ├── GTSRB/
   │   ├── train/
   │   │   ├── 00000/
   │   │   ├── 00001/
   │   │   └── ...  # Each folder contains .ppm images and an Excel file
   ```

3. **Convert Images (Optional)**:  
   If you prefer `.jpg` over `.ppm`, use the provided script to convert:
   ```bash
   python data/convert_ppm_to_jpg.py
   ```

4. **Generate YOLO Annotations**:  
   Parse the Excel files in the dataset and create YOLO `.txt` files:
   ```bash
   python data/generate_annotations_from_excel.py
   ```

5. **Create Train and Validation Lists**:  
   Generate `train.txt` and `val.txt` for YOLOv5:
   ```bash
   python data/generate_image_lists.py
   ```

---

## **Usage**

### **Convert `.ppm` to `.jpg`**
```bash
python data/convert_ppm_to_jpg.py
```
Converts all `.ppm` images in the dataset to `.jpg` format.

---

### **Generate YOLO Annotations**
```bash
python data/generate_annotations_from_excel.py
```
Reads Excel files from each class folder and creates YOLO-style `.txt` annotations.

---

### **Train YOLOv5**
1. Navigate to the `yolov5` directory:
   ```bash
   cd yolov5
   ```

2. Start training:
   ```bash
   python train.py --img 640 --batch 16 --epochs 50        --data ../data/data_config.yaml        --cfg models/yolov5s.yaml        --weights ''        --name gtsrb_detector
   ```

---

### **Train ResNet Classifier**
1. Navigate to the `classification` folder:
   ```bash
   cd classification
   ```

2. Train the ResNet model:
   ```bash
   python train_classifier.py
   ```

---

### **Real-Time Detection and Classification**
1. Run real-time detection and classification:
   ```bash
   python inference/real_time_inference.py
   ```

2. Launch the GUI (optional):
   ```bash
   streamlit run inference/gui_app.py
   ```

---

## **To-Do**

- [x] Dataset processing (Excel to YOLO annotations)
- [x] YOLOv5 training script
- [x] ResNet classifier training
- [x] Real-time detection and classification
- [ ] Fine-tune YOLOv5 and ResNet
- [ ] Add additional GUI features
- [ ] Evaluate performance on a test dataset

---

## **License**

This project is licensed under the MIT License.

---

## **Credits**

- **YOLOv5**: [Ultralytics YOLOv5 GitHub](https://github.com/ultralytics/yolov5)  
- **Dataset**: [GTSRB Dataset](https://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)  
- **PyTorch and Torchvision**: [PyTorch](https://pytorch.org)  
- **Streamlit**: [Streamlit Documentation](https://docs.streamlit.io/)
