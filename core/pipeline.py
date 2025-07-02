import cv2
from config.settings import EMBEDDING_THRESHOLD, DB_PATH, LOGGER_DEBOUNCE_INTERVAL
from db.connection import connect_sqlite
from db.face import FaceDatabase
from db.logger import MatchLoggerDB
from modules.matcher import FaceMatcher
from modules.recognizer import FaceRecognizer
from utils.debouncer import LoggerDebouncer
from utils.fps import FPSCounter
from utils.timer import Timer


def run_pipeline(cap, frame_queue, is_enable_spoof_detection=False):
    # Initialize database connection
    db = connect_sqlite(DB_PATH)
    face_db = FaceDatabase(db)
    logger_db = MatchLoggerDB(db)

    # Initialize face recognizer and matcher
    recognizer = FaceRecognizer(enable_spoof_detection=is_enable_spoof_detection)
    fps = FPSCounter()
    logger_debouncer = LoggerDebouncer(LOGGER_DEBOUNCE_INTERVAL)

    with Timer("Loading embeddings"):
        known_faces = face_db.load_embeddings()

    matcher = FaceMatcher(known_faces, EMBEDDING_THRESHOLD)

    while True:
        if frame_queue.empty():
            continue
        frame = frame_queue.get()

        with Timer("Face detection and embedding"):
            faces = recognizer.detect_and_embed(frame)

        for face in faces:
            with Timer("Face matching"):
                match, score = matcher.match(face.embedding)

            if logger_debouncer.should_log(match, score, EMBEDDING_THRESHOLD):
                with Timer("Save match to database"):
                    logger_db.log(match, score)

            face.draw(frame, match, score)

        fps.update()
        fps.draw(frame)

        cv2.imshow("Real-time Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
