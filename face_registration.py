import cv2
from config.settings import DB_PATH
from db.connection import connect_sqlite
from db.face import FaceDatabase
from modules.recognizer import FaceRecognizer


def register_face():
    cap = cv2.VideoCapture(0)
    recognizer = FaceRecognizer()
    db = connect_sqlite(DB_PATH)
    face_db = FaceDatabase(db)

    print("📸 Press 'c' to capture face, or 'q' to quit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("❌ Exit face registration.")
            break

        elif key == ord("c"):
            faces = recognizer.detect_and_embed(frame)
            if not faces:
                print("⚠️  No face detected. Try again.")
                continue

            if len(faces) > 1:
                print("⚠️  Multiple faces detected. Please show only one.")
                continue

            face = faces[0]
            name = input("Enter name for this face: ").strip()
            if not name:
                print("⚠️  Name cannot be empty.")
                continue

            face_db.insert_face(name, face.embedding)
            print(f"✅ Registered face as '{name}'")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    register_face()
