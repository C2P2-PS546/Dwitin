from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import logging
from logging.handlers import RotatingFileHandler

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler = RotatingFileHandler('ocr_app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

# Load the model
model = load_model('ocr_model.h5')  # Pastikan file 'ocr_model.h5' ada di folder yang sama

# Initialize Flask app
app = Flask(__name__)

# Define character mapping (adjust based on training data)
char_map = "abcdefghijklmnopqrstuvwxyz0123456789 "

# Preprocessing function
def preprocess_image(image):
    """
    Preprocess the input image to match the model's input requirements.
    """
    logging.debug("Starting image preprocessing...")
    try:
        # Convert to grayscale
        image = image.convert('L')

        # Resize to match model input
        image = image.resize((128, 128), Image.BICUBIC)

        # Normalize pixel values
        image = np.array(image, dtype=np.float32) / 255.0

        # Add channel and batch dimensions
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = np.expand_dims(image, axis=0)   # Add batch dimension

        logging.debug(f"Preprocessed image shape: {image.shape}")
        return image

    except Exception as e:
        logging.error(f"Error in preprocessing image: {e}")
        raise



# Decoding predictions
def decode_predictions(predictions):
    """
    Decode raw predictions into meaningful labels: company, address, date, total.
    """
    logging.debug("Starting prediction decoding...")

    try:
        # Decode predictions (Assume CTC decoding)
        decoded = tf.keras.backend.ctc_decode(
            predictions,
            input_length=np.ones(predictions.shape[0]) * predictions.shape[1],
            greedy=True
        )[0][0]

        # Get decoded texts as numpy array
        decoded_texts = tf.keras.backend.get_value(decoded)

        # Define character map
        char_map = "abcdefghijklmnopqrstuvwxyz0123456789 "
        full_text = ''.join([char_map[i] for i in decoded_texts[0] if i != -1])

        logging.debug(f"Decoded text (full): {full_text}")

        # Split decoded text into parts for company, address, date, and total
        labels = ["company", "address", "date", "total"]
        split_points = [15, 30, 40]  # Example split points (adjust as needed)
        
        # Ensure full_text has enough length
        if len(full_text) < split_points[-1]:
            logging.warning(f"Decoded text is too short: {full_text}")
            return {label: "N/A" for label in labels}

        parts = [
            full_text[:split_points[0]],
            full_text[split_points[0]:split_points[1]],
            full_text[split_points[1]:split_points[2]],
            full_text[split_points[2]:],
        ]

        decoded_result = {label: parts[i] if i < len(parts) else "N/A" for i, label in enumerate(labels)}
        return decoded_result

    except Exception as e:
        logging.error(f"Error decoding predictions: {e}")
        return {"company": "N/A", "address": "N/A", "date": "N/A", "total": "N/A"}

def decode_predictions(predictions):
    """
    Decode raw predictions into meaningful labels: company, address, date, total.
    """
    logging.debug("Starting prediction decoding...")

    try:
        # Decode predictions (Assume CTC decoding)
        decoded = tf.keras.backend.ctc_decode(
            predictions,
            input_length=np.ones(predictions.shape[0]) * predictions.shape[1],
            greedy=True
        )[0][0]

        # Get decoded texts as numpy array
        decoded_texts = tf.keras.backend.get_value(decoded)

        # Define character map
        char_map = "abcdefghijklmnopqrstuvwxyz0123456789 "
        full_text = ''.join([char_map[i] for i in decoded_texts[0] if i != -1])

        logging.debug(f"Decoded text (full): {full_text}")

        # Split decoded text into parts for company, address, date, and total
        labels = ["company", "address", "date", "total"]
        split_points = [15, 30, 40]  # Example split points (adjust as needed)

        # Ensure full_text has enough length
        if len(full_text) < split_points[-1]:
            logging.warning(f"Decoded text is too short: {full_text}")
            return {label: "N/A" for label in labels}

        # Dynamically split based on actual length
        parts = []
        last_split = 0
        for point in split_points:
            parts.append(full_text[last_split:point])
            last_split = point
        parts.append(full_text[last_split:])

        decoded_result = {label: parts[i] if i < len(parts) else "N/A" for i, label in enumerate(labels)}
        return decoded_result

    except Exception as e:
        logging.error(f"Error decoding predictions: {e}")
        return {"company": "N/A", "address": "N/A", "date": "N/A", "total": "N/A"}

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        logging.error("No image file in the request.")
        return jsonify({"error": "No image provided."}), 400
    
    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read()))
        logging.debug("Image opened successfully.")
        
        # Preprocess image
        preprocessed_image = preprocess_image(image)
        logging.debug(f"Image preprocessed with shape: {preprocessed_image.shape}")
        
        # Get predictions
        predictions = model.predict(preprocessed_image)
        logging.debug(f"Raw predictions: {predictions}")
        
        # Decode predictions
        decoded_texts = decode_predictions(predictions)
        logging.debug(f"Decoded result: {decoded_texts}")
        
        return jsonify(decoded_texts)
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)