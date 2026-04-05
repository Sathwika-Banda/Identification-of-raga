import streamlit as st
import numpy as np
import joblib
import tempfile
import os
from utils import extract_features

st.set_page_config(
    page_title="Carnatic Raga Identifier",
    page_icon="🎶",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
header[data-testid="stHeader"] {
    background-color: #0a0e27 !important;
    height: 70px !important;
}

header[data-testid="stHeader"]::before {
    content: '🎵 RAGA IDENTIFIER';
    position: absolute;
    left: 20px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 20px;
    font-weight: 900;
    color: #00d4ff;
    letter-spacing: 2px;
}

.stApp {
    background: linear-gradient(135deg, #0a0e27, #1a1f3a);
    font-family: 'Segoe UI', sans-serif;
    color: #eaeaea;
}

.block-container {
    max-width: 100%;
    padding-top: 4.5rem !important;
    padding-left: 5% !important;
    padding-right: 5% !important;
    padding-bottom: 0.5rem !important;
}

[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 15px;
    padding: 8px 15px !important;
    border: 2px solid rgba(0, 212, 255, 0.3);
    box-shadow: 0 10px 40px rgba(0, 212, 255, 0.15);
    margin-bottom: 8px;
    min-height: 40px !important;
}

[data-testid="stFileUploader"] label {
    color: #eaeaea !important;
    font-size: 14px;
    font-weight: 500;
}

.result-card {
    background: linear-gradient(135deg, rgba(15, 52, 96, 0.8), rgba(22, 33, 62, 0.8));
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 15px 40px rgba(0, 212, 255, 0.3);
    text-align: center;
    margin: 8px 0;
    border: 2px solid rgba(0, 212, 255, 0.4);
}

.result-card h2 {
    color: #a8dadc;
    font-size: 14px;
    margin-bottom: 3px;
    font-weight: 400;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.result-card h1 {
    color: #00d4ff;
    font-size: 40px;
    font-weight: 900;
    letter-spacing: 2px;
    text-shadow: 0 0 30px rgba(0, 212, 255, 0.6);
    margin: 3px 0;
}

audio {
    width: 100%;
    margin: 5px 0;
    filter: drop-shadow(0 5px 15px rgba(0, 212, 255, 0.3));
    height: 35px;
}

.stAlert {
    background-color: rgba(255, 193, 7, 0.15) !important;
    border: 1px solid #ffc107 !important;
    color: #ffc107 !important;
    border-radius: 10px !important;
    padding: 8px 12px !important;
    font-size: 12px !important;
    margin: 5px 0 !important;
}

h3 {
    color: #e0aaff;
    font-weight: 600;
    letter-spacing: 1px;
    margin: 5px 0 !important;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

try:
    model = joblib.load("raga_ensemble_model.joblib")
except:
    try:
        model = joblib.load("raga_svm_model.joblib")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

try:
    scaler = joblib.load("raga_scaler.joblib")
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

audio_map = {
    "abheri": "abheri_swarams.wav",
    "bhairavi": "bhairavi_swarams.wav",
    "hamsadwani": "hamsadwani_swarams.wav",
    "hindolam": "hindolam_swarams.wav",
    "kalyani": "kalyani_swarams.wav",
    "malahari": "malahara_swarams.wav",
    "mayamavalagowla": "mayamavalagowla_swarams.wav",
    "mohana": "mohana_swarams.wav",
    "shankarabharanam": "shankarabharanam_swarams.wav",
    "todi": "todi_swarams.wav"
}

uploaded_file = st.file_uploader("Upload Audio", type=["wav","mp3"])

if uploaded_file:
    st.audio(uploaded_file)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    try:
        with st.spinner("Analyzing..."):
            features = extract_features(temp_path)
        
        if features is None:
            st.error("Could not extract features from audio file")
            os.remove(temp_path)
            st.stop()

        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probs = model.predict_proba(features_scaled)[0]
        confidence = max(probs) * 100
        
        prediction_lower = prediction.lower()
        display_raga = prediction.capitalize()

        st.markdown(f"""
        <div class="result-card">
        <h2>Predicted Raga</h2>
        <h1>{display_raga}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        if confidence < 30:
            st.error("⚠️ Low confidence. This might be a raga not in our database.")

        if prediction_lower in audio_map:
            raga_file = audio_map[prediction_lower]
            if os.path.exists(raga_file):
                st.markdown(f'<h3>Listen to {display_raga} Swarams</h3>', unsafe_allow_html=True)
                st.audio(raga_file)
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
