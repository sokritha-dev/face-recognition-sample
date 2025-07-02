from core.pipeline import run_pipeline
from core.video_capture import start_video_capture
from face_registration import register_face


def run_recognition(action):
    if action == "register":
        print("Register face data for recognition...")
        register_face()
    elif action == "test":
        print("Running face recognition...")
        start_video_capture(run_pipeline, is_enable_spoof_detection=False)
