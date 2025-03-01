from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO
import numpy as np
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = os.getenv("MODEL_PATH", "lung_disease_model.keras")

if not os.path.exists(MODEL_PATH):
    app.logger.error(f"Model file not found at {MODEL_PATH}. Check your .env configuration.")
    MODEL_PATH = None

try:
    model = tf.keras.models.load_model(MODEL_PATH) if MODEL_PATH else None
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    model = None

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

if not GEMINI_API_KEY:
    app.logger.error("GEMINI_API_KEY is missing. Check your .env file.")
    GEMINI_API_KEY = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not model:
            return jsonify({'error': 'Model is not loaded properly'}), 500

        # Convert the image
        img = Image.open(BytesIO(file.read())).convert("RGB")
        img = img.resize((150, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize

        prediction = model.predict(img_array)
        prediction_class = np.argmax(prediction, axis=1)[0]
        classes = ["Bacterial Pneumonia", "Corona Virus Disease", "Normal", "Tuberculosis", "Viral Pneumonia"]

        return jsonify({'prediction': classes[prediction_class]})

    except Exception as e:
        return jsonify({'error': 'Failed to process image', 'details': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400

        payload = {"contents": [{"parts": [{"text": user_message}]}]}
        headers = {"Content-Type": "application/json"}
        api_url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

        response = requests.post(api_url_with_key, headers=headers, json=payload)
        response_data = response.json()

        if response.status_code != 200:
            return jsonify({'error': 'Gemini API request failed', 'details': response.text}), response.status_code

        raw_reply = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

        if not raw_reply:
            return jsonify({'error': 'Invalid API response format', 'details': response_data}), 500

        return jsonify({'response': raw_reply.strip()})

    except Exception as e:
        return jsonify({'error': 'Chatbot request failed', 'details': str(e)}), 500

if __name__ == '__main__':
    PORT = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=PORT, debug=True)
