from pipeline import real_time_emotion_recognition, MelSpectrogramPreprocessing, MFCCPreprocessing
from tensorflow.keras.models import load_model

emotion_labels_ravdess = [
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]

if __name__ == "__main__":
    
    # # Keras model loading: cnn-lstm
    model = load_model("models/keras/cnn_lstm_ravdess_mel_spect.keras")
    strategy = MelSpectrogramPreprocessing()
    
    # Keras model loading: cnn-mfcc
    # model = load_model("cnn-ravness-mfcc.keras")
    # strategy = MFCCPreprocessing()
    
    # Start pipeline with chosen preprocessing strategy
    real_time_emotion_recognition(model, strategy=strategy, emotion_labels=emotion_labels_ravdess)