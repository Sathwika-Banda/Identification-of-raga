import streamlit as st
from scripts.preprocess import load_audio, extract_mfcc
import joblib

model = joblib.load('models/raga_classifier.pkl')

st.title("Carnatic Raga Identifier")
audio_file = st.file_uploader("Upload a Carnatic music sample", type=["wav"])

if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    audio = load_audio(audio_file)
    features = extract_mfcc(audio).reshape(1, -1)
    raga = model.predict(features)[0]
    st.success(f"Predicted Raga: {raga}")