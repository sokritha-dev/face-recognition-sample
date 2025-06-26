from insightface.app import FaceAnalysis

class FaceEmbedder:
    def __init__(self):
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0)

    def get_embedding(self, face_img):
        faces = self.app.get(face_img)
        return faces[0].embedding if faces else None
