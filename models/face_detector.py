# models/face_detector.py
import torch.serialization
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv
from torch.nn import Sequential

# ✅ Register trusted classes to allow model loading
torch.serialization.add_safe_globals([DetectionModel, Conv, Sequential])


class FaceDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_faces(self, frame):
        results = self.model.predict(source=frame, conf=0.3, verbose=False)
        return results[0].boxes.xyxy.cpu().numpy() if results else []
