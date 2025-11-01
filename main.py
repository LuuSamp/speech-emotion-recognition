from pipeline import real_time_emotion_recognition, MelSpectrogramPreprocessing, MFCCPreprocessing
from tensorflow.keras.models import load_model


if __name__ == "__main__":
    # ============ CARREGA MODELO ============
    # Para Keras (CNN/LSTM)
    model = load_model("cnn_lstm_ravdess_mel_spect.keras")
    # model = load_model("cnn-ravness-mfcc.keras")
    
    # Inicia pipeline de reconhecimento de emoção em tempo real
    real_time_emotion_recognition(model, strategy=MelSpectrogramPreprocessing())
    # real_time_emotion_recognition(model, strategy=MFCCPreprocessing())