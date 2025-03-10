import cv2
import numpy as np
import tensorflow as tf
from keras.src.saving import load_model
import os
import requests

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
# Google Drive direct download link for model.h5
MODEL_URL = "https://drive.google.com/uc?export=download&id=11UFsJ1SLEi330k7Q2wznZNG5X1UH3JH3"
MODEL_PATH = "./model.h5"

# Download model if it does not exist
if not os.path.exists(MODEL_PATH):
    print("Downloading model.h5 from Google Drive...")
    response = requests.get(MODEL_URL, stream=True)
    
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("Model download complete.")
    else:
        print("Error: Failed to download the model. Check the URL.")
        exit(1)

# Load the model
model = load_model(MODEL_PATH)

def load_video(video_path, frame_size=(64, 64), sequence_length=16):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, frame_size)
        frame = frame / 255.0  # Normalize pixel values
        frames.append(frame)

        if len(frames) == sequence_length:
            break

    cap.release()

    if len(frames) < sequence_length:
        print("Warning: Video is too short, padding with last frame.")
        while len(frames) < sequence_length:
            frames.append(frames[-1])

    return np.array(frames)

def predict_video(video_path):
    frames = load_video(video_path)
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    prediction = model.predict(frames)

    class_labels = ["Non-Violence", "Violence"]
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    return predicted_class, confidence

if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ")
    if not os.path.exists(video_path):
        print("Error: Video file not found!")
        exit(1)

    prediction, confidence = predict_video(video_path)
    print(f"Predicted Class: {prediction} (Confidence: {confidence:.2f})")
