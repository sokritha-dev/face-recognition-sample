from insightface.app import FaceAnalysis
import cv2

from modules.spoofing import AntiSpoofPredictor


class Face:
    """A class representing a detected face with its bounding box and embedding."""

    def __init__(self, bbox, embedding):
        self.bbox = list(map(int, bbox))
        self.embedding = embedding

    def draw(self, frame, name, score, color=(0, 255, 0)):
        x1, y1, x2, y2 = self.bbox
        label = f"{name} | {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )


class FaceRecognizer:
    """A class for recognizing faces using InsightFace, with optional spoof detection."""

    def __init__(self, enable_spoof_detection: bool = False):
        self.app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0)

        self.enable_spoof_detection = enable_spoof_detection
        self.spoof_detector = AntiSpoofPredictor() if enable_spoof_detection else None

    def detect(self, frame):
        return self.app.get(frame)

    def embed(self, raw_face):
        return Face(raw_face.bbox, raw_face.embedding)

    def is_real_face(self, frame, bbox):
        x1, y1, x2, y2 = list(map(int, bbox))
        w, h = x2 - x1, y2 - y1
        # Visualize bounding box for debugging
        # debug_frame = frame.copy()
        # cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.imwrite("debug_bbox.jpg", debug_frame)
        label, confidence = self.spoof_detector.predict(frame, (x1, y1, w, h))
        return label == "real", confidence

    def detect_and_embed(self, frame):
        faces = []
        for raw_face in self.detect(frame):
            if self.enable_spoof_detection:
                # Check if the face is real before embedding
                is_real, confidence = self.is_real_face(frame, raw_face.bbox)
                if not is_real:
                    print(f"\u274c Spoof detected (score={confidence:.2f})")
                    f = Face(raw_face.bbox, None)
                    f.draw(frame, "spoof", confidence, color=(0, 0, 255))
                    continue
            face = self.embed(raw_face)
            faces.append(face)
        return faces
