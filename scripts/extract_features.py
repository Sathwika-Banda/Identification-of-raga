import os
import librosa
import numpy as np
import pandas as pd

# Set paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "features.csv")

# Create processed folder if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Define number of MFCCs to extract
N_MFCC = 20

# Initialize list to hold feature rows
data = []

print("🎵 Extracting features from audio files...")

# Loop through audio files
for filename in os.listdir(RAW_DIR):
    if filename.endswith(".wav"):
        label = filename.split("_")[0]  # Assuming filename like Bhairavi_1.wav
        filepath = os.path.join(RAW_DIR, filename)

        # Load audio
        y, sr = librosa.load(filepath, duration=10)

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc.T, axis=0)  # Shape: (20,)

        # Append to dataset
        data.append(list(mfcc_mean) + [label])

# Create DataFrame
columns = [f"mfcc{i+1}" for i in range(N_MFCC)] + ["label"]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Features extracted and saved to {OUTPUT_FILE}")
