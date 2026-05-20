from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import pandas as pd
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    client = genai.Client(api_key=api_key)
else:
    client = None
    print("WARNING: GEMINI_API_KEY not found in .env file.")

GEMINI_PROMPT = """
You are an expert Agricultural Pathologist and Agronomist. 
Analyze this plant leaf image and identify any diseases, deficiencies, or pests.
Respond ONLY with a valid JSON object using this exact format:
{
    "status": "Healthy" | "Mild Infection" | "Moderate Infection" | "Severe Infection" | "Critical / Dead" | "Not a Leaf",
    "disease_name": "Name of the disease (e.g., Early Blight, Powdery Mildew, No Disease Detected)",
    "confidence": <number between 0-100>,
    "disease_percentage": <number between 0-100>,
    "recommendation_en": "Detailed treatment recommendation and action plan in English.",
    "recommendation_hi": "Detailed treatment recommendation and action plan in Hindi."
}
If the image is completely unreadable or not a plant, return "status": "Not a Leaf" and set percentage to 0.
"""

app = Flask(__name__)
CORS(app)

# Load the Yield Prediction Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'yield_model.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        yield_model = pickle.load(f)
    print("Yield Model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load yield model. Run train_yield.py first. Error: {e}")
    yield_model = None

@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    if not yield_model:
        return jsonify({'error': 'Model not trained yet on server.'}), 500
        
    try:
        data = request.json
        crop = data.get('crop')
        temperature = float(data.get('temperature'))
        rainfall = float(data.get('rainfall'))
        
        if not crop:
            return jsonify({'error': 'Crop is required'}), 400
            
        # Create a DataFrame matching the training data format
        input_data = pd.DataFrame([{
            'Crop': crop,
            'Temperature': temperature,
            'Rainfall': rainfall
        }])
        
        prediction = yield_model.predict(input_data)[0]
        
        return jsonify({
            'success': True,
            'crop': crop,
            'predicted_yield_q_per_ha': round(prediction, 2),
            'predicted_yield_kg_per_acre': round(prediction * 100 * 0.404686, 2) # Conversion
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/detect_disease', methods=['POST'])
def detect_disease():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    try:
        file = request.files['image']
        img_bytes = file.read()
        
        if not client:
            return jsonify({'error': 'GEMINI_API_KEY is not configured in the server .env file.'}), 500
            
        # Format the image for Gemini using new SDK
        image_part = types.Part.from_bytes(
            data=img_bytes,
            mime_type=file.mimetype or "image/jpeg"
        )
        
        # Call Gemini 2.5 Flash Vision Model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[GEMINI_PROMPT, image_part]
        )
        
        # Parse the JSON response
        text_res = response.text
        
        # Clean markdown code blocks if Gemini added them
        if "```json" in text_res:
            text_res = text_res.split("```json")[1].split("```")[0].strip()
        elif "```" in text_res:
            text_res = text_res.split("```")[1].strip()
            
        result = json.loads(text_res)
        result['success'] = True
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in Gemini Vision AI: {str(e)}")
        return jsonify({'error': f"AI Analysis Failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting Smart Yield ML Microservice on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
