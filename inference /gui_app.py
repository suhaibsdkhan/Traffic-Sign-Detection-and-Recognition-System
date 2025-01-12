# ======================
# 6) gui_app.py
# ======================

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

@st.cache_resource
def load_models(yolo_weights_path, resnet_weights_path, device, num_classes):
    yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights_path, force_reload=False)
    yolo.to(device)
    yolo.eval()
    rnet = models.resnet18(pretrained=False)
    rnet.fc = nn.Linear(rnet.fc.in_features, num_classes)
    rnet.load_state_dict(torch.load(resnet_weights_path, map_location=device))
    rnet.to(device)
    rnet.eval()
    return yolo, rnet

def main():
    st.title("Traffic Sign Detection & Recognition")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_weights_path = "/path/to/best.pt"
    resnet_weights_path = "/path/to/resnet_gtsrb.pth"

    class_names = [
        "Speed limit (20km/h)",
        "Speed limit (30km/h)",
        "Speed limit (50km/h)",
        "Speed limit (60km/h)",
        "Speed limit (70km/h)",
        "Speed limit (80km/h)",
        "End of speed limit (80km/h)",
        "Speed limit (100km/h)",
        "Speed limit (120km/h)",
        "No passing",
        "No passing for vehicles over 3.5 metric tons",
        "Right-of-way at the next intersection",
        "Priority road",
        "Give way",
        "Stop",
        "No vehicles",
        "Vehicles over 3.5 metric tons prohibited",
        "No entry",
        "General caution",
        "Dangerous curve to the left",
        "Dangerous curve to the right",
        "Double curve",
        "Bumpy road",
        "Slippery road",
        "Road narrows on the right",
        "Road work",
        "Traffic signals",
        "Pedestrians",
        "Children crossing",
        "Bicycles crossing",
        "Beware of ice/snow",
        "Wild animals crossing",
        "End of all speed and passing limits",
        "Turn right ahead",
        "Turn left ahead",
        "Ahead only",
        "Go straight or right",
        "Go straight or left",
        "Keep right",
        "Keep left",
        "Roundabout mandatory",
        "End of no passing",
        "End of no passing by vehicles over 3.5 metric tons"
    ]

    yolo_model, resnet_model = load_models(yolo_weights_path, resnet_weights_path, device, len(class_names))

    transform_pipeline = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            results = yolo_model(frame_rgb)
        det = results.xyxy[0].cpu().numpy()
        for *bbox, conf, cls_id in det:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            timg = transform_pipeline(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = resnet_model(timg)
                _, pred_cls = torch.max(out, 1)
            pid = pred_cls.item()
            lbl = class_names[pid] if pid < len(class_names) else f"Class {pid}"
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame_rgb, f"{lbl} ({conf:.2f})", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        st.image(frame_rgb, channels="RGB")

if __name__ == '__main__':
    main()