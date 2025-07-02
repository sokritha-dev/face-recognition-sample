from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2

from models.mini_fasnet import (
    MiniFASNetV1SE,
)  # Make sure this matches your fine-tuned model architecture


class AntiSpoofPredictor:
    def __init__(self, model_path="minifasnet_finetuned.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Load your fine-tuned model
        self.model = MiniFASNetV1SE(
            img_channel=3, num_classes=2, embedding_size=128, conv6_kernel=(5, 5)
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Label map
        self.label_map = {0: "fake", 1: "real", 2: "uncertain"}

    def preprocess(self, img, bbox, scale=2.5):
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        size = int(max(w, h) * scale)

        left = max(cx - size // 2, 0)
        top = max(cy - size // 2, 0)
        right = min(cx + size // 2, img.shape[1])
        bottom = min(cy + size // 2, img.shape[0])

        cropped = img[top:bottom, left:right]
        resized = cv2.resize(cropped, (80, 80))
        tensor = self.transform(resized).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, img: np.ndarray, bbox: tuple):
        face = self.preprocess(img, bbox)

        with torch.no_grad():
            softmax_output = F.softmax(self.model(face), dim=1).cpu().numpy()[0]

        print("Softmax scores [fake, real, uncertain]:", softmax_output)
        if softmax_output[1] > 0.7:
            label = 1
        else:
            label = 0

        confidence = float(softmax_output[label])

        print(f"Predicted: {self.label_map[label]} (confidence={confidence:.2f})")
        return self.label_map[label], confidence
