import pyaudio
import webrtcvad
import numpy as np
from collections import deque
import tensorflow as tf
from .preprocessing import PreprocessingStrategy

# ============ CONFIGURAÇÕES ============
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 10  # chunk duration in ms
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
BUFFER_DURATION_SEC = 3  # buffer duration for classification (3 seconds)
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION_SEC)

emotion_labels = [
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]

# Energy threshold to filter out low-energy segments
energy_threshold = 0.003

def is_high_energy(audio_segment, thresh=energy_threshold):
    energy = np.sqrt(np.mean(np.square(audio_segment)))
    return energy > thresh

# VAD setup
vad = webrtcvad.Vad(2)

audio_buffer = deque(maxlen=BUFFER_SIZE)

def classify_emotion(model, audio_data, strategy: PreprocessingStrategy = None):
    """Classify emotion from audio data using the provided model and preprocessing strategy."""
    if strategy:
        x = strategy.preprocess(audio_data)
    else:
        x = audio_data
        
    # Predição
    prediction = model.predict(x, verbose=0)
    emotion_idx = np.argmax(prediction)
    
    emotion = emotion_labels[emotion_idx]
    confidence = prediction[0][emotion_idx]
    
    return emotion, confidence


def real_time_emotion_recognition(model, strategy: PreprocessingStrategy = None):
    """Real-time emotion recognition pipeline"""
    p = pyaudio.PyAudio()

    # Open microphone stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    print("🎤 Listening... Say Something!")
    
    try:
        speech_detected_said = False
        while True:
            is_speech = False
            # Read audio chunk from microphone
            raw_audio = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            # Convert bytes to numpy array
            audio_chunk = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Voice Activity Detection
            is_speech = vad.is_speech(raw_audio, SAMPLE_RATE)
            
            if( is_speech and is_high_energy(audio_chunk) ):
                # Add to circular buffer
                audio_buffer.extend(audio_chunk)
            
            
            if is_speech and len(audio_buffer) >= BUFFER_SIZE:
                if not speech_detected_said:
                    print("🗣️  Speech detected! Classifying emotion...")
                    speech_detected_said = True
                else:
                    # Clear console line
                    print("\033[A\033[K", end="")
                    print("\033[A\033[K", end="")
                
                # Get last N seconds from buffer
                audio_segment = np.array(list(audio_buffer)[-BUFFER_SIZE:])
                
                # Classify emotion
                emotion, confidence = classify_emotion(model, audio_segment, strategy)
                
                print(f"✅ Emotion detected: {emotion} (confidence: {confidence:.2%})")
                print("-" * 50)
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        tf.keras.backend.clear_session()