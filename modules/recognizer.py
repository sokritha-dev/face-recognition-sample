from insightface.app import FaceAnalysis
import cv2


class Face:
    """A class representing a detected face with its bounding box and embedding."""

    def __init__(self, bbox, embedding):
        self.bbox = list(map(int, bbox))
        self.embedding = embedding

    def draw(self, frame, name, score):
        x1, y1, x2, y2 = self.bbox
        label = f"{name} | {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )


class FaceRecognizer:
    """A class for recognizing faces using InsightFace."""

    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0)

    def detect(self, frame):
        return self.app.get(frame)  # returns InsightFace objects

    def embed(self, raw_face):
        return Face(raw_face.bbox, raw_face.embedding)  # returns a Face embedding

    def detect_and_embed(self, frame):
        return [self.embed(f) for f in self.detect(frame)]
