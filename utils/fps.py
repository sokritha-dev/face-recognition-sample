import time
import cv2


class FPSCounter:
    """A class to calculate and display frames per second (FPS) in a video stream."""

    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0.0

    def update(self):
        curr_time = time.time()
        self.fps = 1.0 / (curr_time - self.prev_time)
        self.prev_time = curr_time

    def draw(self, frame):
        cv2.putText(
            frame,
            f"FPS: {self.fps:.2f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
