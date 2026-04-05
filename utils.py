import librosa
import numpy as np

def extract_features(file_path):
    """Unified feature extraction - 79 features total"""
    try:
        y, sr = librosa.load(file_path, duration=30)
        y, _ = librosa.effects.trim(y)
        y = librosa.util.normalize(y)

        # MFCC: 20 mean + 20 std = 40
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)

        # Chroma: 12 mean + 12 std = 24
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        chroma_std = np.std(chroma.T, axis=0)

        # Spectral Contrast: 7
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)

        # Tonnetz: 6
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        tonnetz_mean = np.mean(tonnetz.T, axis=0)

        features = np.hstack([mfccs_mean, mfccs_std, chroma_mean, chroma_std, contrast_mean, tonnetz_mean])
        
        if np.isnan(features).any():
            return None
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None
