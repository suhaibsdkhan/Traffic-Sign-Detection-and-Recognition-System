# ======================
# 5) real_time_inference.py
# ======================

import cv2
import torch
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

def load_yolo_model(yolo_weights, device):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weights, force_reload=False)
    model.to(device)
    model.eval()
    return model

def load_resnet_classifier(resnet_weights, device, num_classes):
    m = models.resnet18(pretrained=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.load_state_dict(torch.load(resnet_weights, map_location=device))
    m.to(device)
    m.eval()
    return m

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model = load_yolo_model('/path/to/best.pt', device)
    resnet_model = load_resnet_classifier('/path/to/resnet_gtsrb.pth', device, 43)

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

    transform_pipeline = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)
        det = results.xyxy[0].cpu().numpy()
        for *bbox, conf, cls_id in det:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor_img = transform_pipeline(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = resnet_model(tensor_img)
                _, pred_class = torch.max(out, 1)
            cidx = pred_class.item()
            lbl = class_names[cidx] if cidx < len(class_names) else f"Class {cidx}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{lbl} ({conf:.2f})", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("Traffic Sign Detection & Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()