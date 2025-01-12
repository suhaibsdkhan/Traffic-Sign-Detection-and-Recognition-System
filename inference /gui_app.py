import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

@st.cache_resource
def load_models(yolo_weights_path, resnet_weights_path, device, num_classes=43):
    yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights_path, force_reload=False)
    yolo.to(device)
    yolo.eval()

    rnet = models.resnet18(pretrained=False)
    rnet.fc = nn.Linear(rnet.fc.in_features, num_classes)
    rnet.load_state_dict(torch.load(resnet_weights_path, map_location=device))
    rnet.to(device)
    rnet.eval()

    return yolo, rnet

def classify_image(classifier, img_np, device, transform, class_names):
    pil_img = Image.fromarray(img_np)
    t = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = classifier(t)
        _, pred = torch.max(out, 1)
    idx = pred.item()
    return class_names[idx] if idx < len(class_names) else f"Class {idx}"

def main():
    st.title("Traffic Sign Detection & Recognition")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo_weights_path = "./yolov5/runs/train/gtsrb_detector/weights/best.pt"
    resnet_weights_path = "./classification/models/resnet_gtsrb.pth"
    class_names = [f"Class {i}" for i in range(43)]

    st.write("Loading models...")
    yolo_model, resnet_model = load_models(yolo_weights_path, resnet_weights_path, device, len(class_names))
    st.write("Models loaded.")

    transform_pipeline = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg","jpeg","png","ppm"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, 1)  # BGR
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            results = yolo_model(frame_rgb)
        det = results.xyxy[0].cpu().numpy()

        for *bbox, conf, cls_id in det:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            label = classify_image(resnet_model, crop_rgb, device, transform_pipeline, class_names)

            cv2.rectangle(frame_rgb, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame_rgb, f"{label} ({conf:.2f})", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        st.image(frame_rgb, channels="RGB", caption="Detection & Classification Results")

if __name__ == "__main__":
    main()