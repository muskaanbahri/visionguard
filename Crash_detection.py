import cv2
import numpy as np
import tensorflow as tf
from keras.src.saving import load_model
import os

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

def predict_video(model, video_path):
    frames = load_video(video_path)
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    prediction = model.predict(frames)

    class_labels = ["Non-Violence", "Violence"]
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    return predicted_class, confidence

if __name__ == "__main__":
    model_path = "./model.h5"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        exit(1)

    model = load_model(model_path)

    video_path = input("Enter the path to the video file: ")
    if not os.path.exists(video_path):
        print("Error: Video file not found!")
        exit(1)

    prediction, confidence = predict_video(model, video_path)
    print(f"Predicted Class: {prediction} (Confidence: {confidence:.2f})")
