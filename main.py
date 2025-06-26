import cv2
import time
from config.settings import MODEL_PATH, DB_PATH, EMBEDDING_THRESHOLD
from models.face_detector import FaceDetector
from models.face_embedder import FaceEmbedder
from utils.similarity import cosine_similarity
from db.face import connect_db, load_embeddings
from db.logger import setup_logger, log_match

detector = FaceDetector(MODEL_PATH)
embedder = FaceEmbedder()
conn = connect_db(DB_PATH)
setup_logger(conn)
known_faces = load_embeddings(conn)

cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes = detector.detect_faces(frame)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face_crop = frame[y1:y2, x1:x2]
        embedding = embedder.get_embedding(face_crop)
        if embedding is None:
            continue

        best_match, best_score = "Unknown", 0.0
        for _, name, known_embedding in known_faces:
            score = cosine_similarity(embedding, known_embedding)
            if score > best_score and score > EMBEDDING_THRESHOLD:
                best_score = score
                best_match = name

        # ✅ Log to SQLite
        log_match(conn, best_match, best_score)

        # ✅ Draw box and info
        label = f"{best_match} | Confidence: {best_score:.2f}"
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

    # ✅ Calculate and display FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    # ✅ Show frame
    cv2.imshow("Real-time Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
