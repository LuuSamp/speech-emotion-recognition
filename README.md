# Speech Emotion Recognition

Real-time speech emotion recognition using pre-trained Keras models and reusable audio preprocessing strategies (MFCC and Mel-spectrogram). Done as a deep learning project to explore audio feature extraction and model architectures for emotion classification from speech.

The models used were trained on the RAVDESS dataset and are intended for experimentation only. The training was done by the notebooks in the `notebooks/` directory, and the datasets used, as well as the tutorial followed, are referenced there.

## What the project does

This repository provides a small, easy-to-run pipeline that captures microphone audio, detects speech with WebRTC VAD, extracts features (MFCC or framed Mel-spectrogram), and performs emotion classification using Keras models. The realtime loop and preprocessing strategies are implemented in `pipeline/`. You can add your own trained Keras model files and preprocessing strategies to run real-time emotion recognition from live audio input.

The example models provided were trained on the RAVDESS dataset using notebooks in `notebooks/`. They are intended for experimentation, and are too unreliable and resource-intensive for actual use cases.

## Why this is useful

- Quickly prototype and demo emotion recognition from live audio.
- Reusable preprocessing classes (`MFCCPreprocessing`, `MelSpectrogramPreprocessing`) simplify feature extraction for training and inference.
- Example notebooks and included model files (if present) help reproduce experiments.

## Repository layout

- `main.py` — example entry point that loads a model and runs the real-time demo
- `pipeline/`
  - `pipeline/pipeline.py` — microphone capture, VAD, buffering and inference loop
  - `pipeline/preprocessing.py` — `PreprocessingStrategy`, `MelSpectrogramPreprocessing`, `MFCCPreprocessing`
- `requirements.txt` — Python dependencies
- `notebooks/` — experiments and training exploration. Datasets used are mentioned at the start of the notebooks(e.g. `cnn-lstm.ipynb`)
- `models/` — example pre-trained model files (may be present in repo root)

## Quick start

Requirements: Python 3.8+ and a working microphone.

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Ensure a model file is available in the repository root or update `main.py` to the model path. `main.py` uses these example names:

- `models/keras/cnn-ravness-mfcc.keras` (MFCC model)
- `models/keras/cnn_lstm_ravdess_mel_spect.keras` (Mel-spectrogram / CNN-LSTM model)

4. Run the real-time demo:

```bash
python main.py
```

`main.py` loads the MFCC model by default and starts a real-time loop that prints the detected emotion and confidence. To switch models or preprocessing strategies, edit `main.py` or import the pipeline functions into your own script.

### Programmatic example

```python
from tensorflow.keras.models import load_model
from pipeline import real_time_emotion_recognition, MFCCPreprocessing

model = load_model("models/keras/cnn-ravness-mfcc.keras")
strategy = MFCCPreprocessing()
real_time_emotion_recognition(model, strategy=strategy)
```

## Preprocessing strategies

- `MFCCPreprocessing`: pads/truncates audio to `max_length_sec`, computes MFCCs and returns a tensor shaped for the MFCC model.
- `MelSpectrogramPreprocessing`: computes a mel-spectrogram, splits it into temporal frames, pads/truncates to 5 frames and returns an input shaped for CNN-LSTM models (1, 5, 128, 128, 1).

Tune preprocessing parameters in `pipeline/preprocessing.py` constructors (sample rate, n_mfcc, window sizes).

## Models & datasets

Place trained Keras model files in `models/keras/` or update `main.py`/your script to point to their location.

Use notebooks in `notebooks/` to check the training of the example models and experimentation.
