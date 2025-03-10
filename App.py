from flask import Flask, request, jsonify
from flask_cors import CORS
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from werkzeug.utils import secure_filename
from Crash_detection import predict_video, load_model

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Ensure the model file path is correct
model_path = "./model.h5"  # Update path if necessary
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = load_model(model_path)

# Ensure upload directory exists
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file temporarily
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Predict the video
    prediction, confidence = predict_video(model, file_path)

    # Clean up the temporary file
    os.remove(file_path)

    return jsonify({
        'class': prediction,
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True)
