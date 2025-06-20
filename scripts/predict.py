import sys
import joblib
import numpy as np
from preprocess import load_audio, extract_mfcc

# Load trained model
model = joblib.load("models/raga_classifier.pkl")

# Load and preprocess input audio
file_path = sys.argv[1]
audio = load_audio(file_path)

# Extract MFCCs and take mean across time axis
mfcc = extract_mfcc(audio, n_mfcc=5)
mfcc_mean = np.mean(mfcc, axis=1)  # shape = (5,)

# Reshape for model prediction
features = mfcc_mean.reshape(1, -1)  # shape = (1, 5)

# Predict
prediction = model.predict(features)
print("Predicted Raga:", prediction[0])
