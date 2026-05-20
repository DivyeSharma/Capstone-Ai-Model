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

# ==============================
# Load Environment Variables
# ==============================
load_dotenv()

# ==============================
# Flask App Setup
# ==============================
app = Flask(__name__)
CORS(app)

# ==============================
# Configure Gemini API
# ==============================
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    client = genai.Client(api_key=api_key)
    print("Gemini API Connected Successfully")
else:
    client = None
    print("WARNING: GEMINI_API_KEY not found")

# ==============================
# Gemini Prompt
# ==============================
GEMINI_PROMPT = """
You are an expert Agricultural Pathologist and Agronomist.

Analyze this plant leaf image and identify:
- diseases
- nutrient deficiencies
- pests

Respond ONLY in valid JSON format:

{
    "status": "Healthy" | "Mild Infection" | "Moderate Infection" | "Severe Infection" | "Critical / Dead" | "Not a Leaf",
    "disease_name": "Disease Name",
    "confidence": 0,
    "disease_percentage": 0,
    "recommendation_en": "English recommendation",
    "recommendation_hi": "Hindi recommendation"
}

If image is not a leaf return:
{
    "status": "Not a Leaf"
}
"""

# ==============================
# Load Yield Prediction Model
# ==============================
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "yield_model.pkl"
)

try:
    with open(MODEL_PATH, "rb") as file:
        yield_model = pickle.load(file)

    print("Yield Model Loaded Successfully")

except Exception as e:
    print(f"Model Loading Error: {e}")
    yield_model = None

# ==============================
# Home Route
# ==============================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "Smart Crop Advisory AI Backend Running"
    })

# ==============================
# Health Check Route
# ==============================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })

# ==============================
# Yield Prediction Route
# ==============================
@app.route("/predict_yield", methods=["POST"])
def predict_yield():

    if yield_model is None:
        return jsonify({
            "success": False,
            "error": "Yield model not loaded"
        }), 500

    try:
        data = request.get_json()

        crop = data.get("crop")
        temperature = float(data.get("temperature"))
        rainfall = float(data.get("rainfall"))

        if not crop:
            return jsonify({
                "success": False,
                "error": "Crop is required"
            }), 400

        # Create dataframe
        input_df = pd.DataFrame([{
            "Crop": crop,
            "Temperature": temperature,
            "Rainfall": rainfall
        }])

        # Prediction
        prediction = yield_model.predict(input_df)[0]

        return jsonify({
            "success": True,
            "crop": crop,
            "predicted_yield_q_per_ha": round(float(prediction), 2),
            "predicted_yield_kg_per_acre": round(
                float(prediction) * 100 * 0.404686,
                2
            )
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

# ==============================
# Disease Detection Route
# ==============================
@app.route("/detect_disease", methods=["POST"])
def detect_disease():

    try:

        # Check image
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image uploaded"
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No selected image"
            }), 400

        # Gemini API check
        if client is None:
            return jsonify({
                "success": False,
                "error": "Gemini API Key not configured"
            }), 500

        # Read image bytes
        img_bytes = file.read()

        # Convert image for Gemini
        image_part = types.Part.from_bytes(
            data=img_bytes,
            mime_type=file.mimetype or "image/jpeg"
        )

        # Generate AI response
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                GEMINI_PROMPT,
                image_part
            ]
        )

        text_res = response.text.strip()

        # Remove markdown if present
        if "```json" in text_res:
            text_res = text_res.split("```json")[1].split("```")[0].strip()

        elif "```" in text_res:
            text_res = text_res.split("```")[1].strip()

        # Parse JSON
        result = json.loads(text_res)

        result["success"] = True

        return jsonify(result)

    except Exception as e:

        print(f"Disease Detection Error: {e}")

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ==============================
# Main
# ==============================
if __name__ == "__main__":

    PORT = int(os.environ.get("PORT", 5000))

    print(f"Server Running on Port {PORT}")

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False
    )
