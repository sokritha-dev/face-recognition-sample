from utils.similarity import cosine_similarity


class FaceMatcher:
    """Class to match face embeddings against known faces using cosine similarity."""

    def __init__(self, known_faces, threshold):
        self.known_faces = known_faces
        self.threshold = threshold

    def match(self, embedding):
        best_match, best_score = "Unknown", 0.0
        for _, name, known_emb in self.known_faces:
            score = cosine_similarity(embedding, known_emb)
            if score > best_score and score > self.threshold:
                best_score = score
                best_match = name
        return best_match, best_score
