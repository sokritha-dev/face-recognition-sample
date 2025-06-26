# add_face.py
import cv2
from config.settings import DB_PATH
from db.face import connect_db, insert_face
from models.face_embedder import FaceEmbedder

name = input("Enter your name: ")
embedder = FaceEmbedder()
conn = connect_db(DB_PATH)

cap = cv2.VideoCapture(0)
print("Show your face to the webcam and press SPACE to capture")

while True:
    ret, frame = cap.read()
    if not ret: break

    cv2.imshow("Add Face", frame)
    key = cv2.waitKey(1)

    if key == 32:  # SPACE key
        embedding = embedder.get_embedding(frame)
        if embedding is not None:
            insert_face(conn, name, embedding)
            print(f"[✓] Added {name} to DB")
        else:
            print("[x] No face detected!")
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
