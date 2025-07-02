import cv2
import threading
import queue

frame_queue = queue.Queue(maxsize=1)


def capture_loop(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)


def start_video_capture(process_fn):
    cap = cv2.VideoCapture(0)
    cap.set(
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
    )  # Try forcing MJPG for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Frame width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Frame height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        raise RuntimeError("Could not open video capture")

    # Start the capture loop in a separate thread, with daemon=True to ensure it exits when the main thread does
    threading.Thread(target=capture_loop, args=(cap,), daemon=True).start()
    process_fn(cap, frame_queue)
