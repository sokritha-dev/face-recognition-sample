import cv2
import os

# Output folders
real_dir = "dataset/real"
fake_dir = "dataset/fake"
os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

print("Press 'r' to save REAL sample, 'f' for FAKE, 'q' to quit.")

img_count = {"real": len(os.listdir(real_dir)), "fake": len(os.listdir(fake_dir))}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw instructions
    cv2.putText(
        frame,
        "Press 'r' to save REAL, 'f' for FAKE, 'q' to quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Dataset Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        filename = os.path.join(real_dir, f"real_{img_count['real']:04d}.jpg")
        cv2.imwrite(filename, frame)
        img_count["real"] += 1
        print(f"✅ Saved REAL: {filename}")

    elif key == ord("f"):
        filename = os.path.join(fake_dir, f"fake_{img_count['fake']:04d}.jpg")
        cv2.imwrite(filename, frame)
        img_count["fake"] += 1
        print(f"✅ Saved FAKE: {filename}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
