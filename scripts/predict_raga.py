import sys
import numpy as np
import librosa
import joblib
import os

# === CONFIGURATION ===
N_MFCC = 20  # Must match extract_features.py

# === LOAD TRAINED MODEL ===
MODEL_PATH = "models/raga_classifier.pkl"
if not os.path.exists(MODEL_PATH):
    print("❌ Trained model not found! Please run train_model.py first.")
    sys.exit(1)

model = joblib.load(MODEL_PATH)

# === LOAD AUDIO INPUT ===
if len(sys.argv) != 2:
    print("Usage: python scripts/predict_raga.py <path_to_audio.wav>")
    sys.exit(1)

audio_path = sys.argv[1]
if not os.path.exists(audio_path):
    print(f"❌ Audio file not found: {audio_path}")
    sys.exit(1)

# === FEATURE EXTRACTION ===
y, sr = librosa.load(audio_path, duration=10)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
mfcc_mean = np.mean(mfcc.T, axis=0)

# === PREDICT RAGA ===
features = mfcc_mean.reshape(1, -1)
prediction = model.predict(features)[0]

print(f"🎵 Predicted Raga: {prediction}")
