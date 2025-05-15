import librosa
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import gradio as gr

# --- Load Trained Model and Scaler ---
model = joblib.load("best_random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

# --- Feature Extraction ---
def extract_features_from_audio(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=22050, duration=30)
        if len(y) < sr * 5:
            return None

        segments = [y[0:sr * 10], y[sr * 10:sr * 20], y[sr * 20:sr * 30]]
        all_features = []

        for segment in segments:
            if np.sum(np.abs(segment)) < 0.01:
                continue

            mfcc = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(y=segment, sr=sr).T, axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr).T, axis=0)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=segment).T, axis=0).flatten()
            onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
            tempo = np.mean(onset_env)

            if not (mfcc.ndim == chroma.ndim == contrast.ndim == zcr.ndim == 1):
                continue

            features = np.hstack([mfcc, chroma, contrast, zcr, [tempo]])
            all_features.append(features)

        if len(all_features) == 0:
            return None

        return np.mean(all_features, axis=0)

    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None

# --- Prediction Function ---
def predict_genre(audio_file):
    features = extract_features_from_audio(audio_file)
    if features is None:
        return "Could not extract features from this audio."
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    genre = encoder.inverse_transform(prediction)[0]
    return f"Predicted Genre: {genre}"

# --- Gradio Interface ---
interface = gr.Interface(
    fn=predict_genre,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="🎵 Chat-Based Music Genre Classifier",
    description="Upload a music clip (.wav) and get the predicted genre."
)

interface.launch()
