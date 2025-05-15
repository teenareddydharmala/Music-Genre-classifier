# 🎵 Music Genre Classification

# Chat-Based Music Genre Classifier

This project is a machine learning-based system that classifies `.wav` audio files into music genres using a trained Random Forest classifier. It also includes a Gradio-powered web interface that allows users to upload music files and receive real-time genre predictions through a chat-like interface.

## Features
- **Audio Feature Extraction**: MFCC, Chroma, Spectral Contrast, Zero Crossing Rate, and Tempo using Librosa.
- **Model**: Random Forest Classifier with hyperparameter tuning using GridSearchCV and StratifiedKFold.
- **Interface**: Chat-based prediction using Gradio's audio upload and response system.
- **Evaluation**: Classification report, confusion matrix, and feature importance visualization.

# Dataset

The GTZAN Genre Collection is one of the most widely used benchmark datasets for music genre classification. It contains 1,000 audio tracks, each 30 seconds long, across 10 distinct genres.


## You can find the dataset via the link below

 https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification


## 📚 Tools Used
- Python
- Librosa
- Gradio
- Joblib
- Scikit-learn
- Matplotlib

## 📂 Structure
- `notebooks/`: All Jupyter notebooks
- `data/`: Dataset 


## 🚀 Run This Project
1. Clone the repo

2. Install packages:
pip install -r requirements.txt

3. Run the chat interface in terminal

python chat_based_music_classifier.py


