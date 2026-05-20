import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

print("Generating simulated agricultural dataset...")

# Define some crops and their ideal conditions (temp, rain) and base yield (quintals/hectare)
crop_data = {
    'Wheat': {'temp': 20, 'rain': 50, 'base_yield': 35},
    'Rice': {'temp': 28, 'rain': 150, 'base_yield': 40},
    'Maize': {'temp': 25, 'rain': 80, 'base_yield': 30},
    'Sugarcane': {'temp': 30, 'rain': 200, 'base_yield': 700},
    'Cotton': {'temp': 28, 'rain': 100, 'base_yield': 20},
    'Soybean': {'temp': 26, 'rain': 90, 'base_yield': 25},
    'Mustard': {'temp': 18, 'rain': 40, 'base_yield': 15},
    'Millets': {'temp': 32, 'rain': 30, 'base_yield': 18},
    'Jowar': {'temp': 30, 'rain': 50, 'base_yield': 22},
    'Groundnut': {'temp': 27, 'rain': 60, 'base_yield': 15},
    'Tomato': {'temp': 24, 'rain': 80, 'base_yield': 250},
    'Potato': {'temp': 18, 'rain': 60, 'base_yield': 200},
    'Onion': {'temp': 22, 'rain': 50, 'base_yield': 150}
}

# Generate 5000 random samples
np.random.seed(42)
data = []

crops = list(crop_data.keys())

for _ in range(5000):
    crop = np.random.choice(crops)
    ideal = crop_data[crop]
    
    # Add random noise to temp and rainfall
    temp = np.random.normal(ideal['temp'], 5)
    rain = np.random.normal(ideal['rain'], 30)
    
    # Ensure realistic values
    temp = max(5, min(temp, 45))
    rain = max(0, rain)
    
    # Calculate yield based on how close conditions are to ideal
    temp_penalty = abs(temp - ideal['temp']) / ideal['temp']
    rain_penalty = abs(rain - ideal['rain']) / ideal['rain']
    
    # Yield reduces if conditions are far from ideal
    yield_multiplier = max(0.1, 1.0 - (temp_penalty * 0.5) - (rain_penalty * 0.5))
    
    # Add some random noise to the yield
    actual_yield = ideal['base_yield'] * yield_multiplier * np.random.normal(1.0, 0.1)
    actual_yield = max(1.0, actual_yield)
    
    data.append({
        'Crop': crop,
        'Temperature': round(temp, 1),
        'Rainfall': round(rain, 1),
        'Yield': round(actual_yield, 2)
    })

df = pd.DataFrame(data)

print(f"Dataset generated with {len(df)} samples.")
print(df.head())

# Prepare features and target
X = df[['Crop', 'Temperature', 'Rainfall']]
y = df['Yield']

# Create a preprocessing pipeline
# One-hot encode the categorical 'Crop' column
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Crop'])
    ],
    remainder='passthrough' # Leave Temperature and Rainfall as they are
)

# Create the final pipeline with Random Forest
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

print("Training Random Forest Regressor...")
model.fit(X, y)

score = model.score(X, y)
print(f"Model trained! R^2 Score on training data: {score:.4f}")

# Save the model
model_path = os.path.join(os.path.dirname(__file__), 'yield_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved successfully to {model_path}")
