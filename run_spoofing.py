from collect_dataset import collect_real_fake_dataset
from core.pipeline import run_pipeline
from core.video_capture import start_video_capture
from face_registration import register_face
from trainers.anti_spoofing_trainer import train_anti_spoof_model


def run_spoofing(action):
    if action == "register":
        print("Register face data for recognition...")
        register_face()
    elif action == "capture":
        print("Capturing real and fakeface data for recognition with spoofing...")
        collect_real_fake_dataset()
        train_anti_spoof_model()
    elif action == "test":
        print("Running face recognition with spoofing...")
        start_video_capture(run_pipeline, is_enable_spoof_detection=True)
