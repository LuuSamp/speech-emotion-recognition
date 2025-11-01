from abc import ABC, abstractmethod
import librosa
import numpy as np

class PreprocessingStrategy(ABC):
    @abstractmethod
    def preprocess(self, audio_data):
        pass


class MelSpectrogramPreprocessing(PreprocessingStrategy):
    def __init__(self, window_size=128, gap=64, sr=16000, n_fft=512, win_length=256,
                 hop_length=128, window='hamming', n_mels=128, fmax=4000):
        
        self._mel_spec_params = {
            'sr': sr,
            'n_fft': n_fft,
            'win_length': win_length,
            'hop_length': hop_length,
            'window': window,
            'n_mels': n_mels,
            'fmax': fmax
        }
        self._frame_mel_params = {
            'window_size': window_size,
            'gap': gap
        }

    def _mel_spectrogram(self, y, sr=16000, n_fft=512, win_length=256,
                         hop_length=128, window='hamming', n_mels=128, fmax=4000):
        mel_feat = np.abs(librosa.stft(y, n_fft=n_fft, window=window,
                                       win_length=win_length, hop_length=hop_length)) ** 2
        mel_feat = librosa.feature.melspectrogram(S=mel_feat, sr=sr, n_mels=n_mels, fmax=fmax)
        mel_feat = librosa.power_to_db(mel_feat, ref=np.max)
        return mel_feat
    
    def _frame_mel_spectrogram(self, mel_spec, window_size=128, gap=64):
        """
        Segmenta Mel Spectrogram em frames temporais (como no treinamento)
        """
        n_frames = (mel_spec.shape[1] - window_size) // gap + 1
        
        if n_frames <= 0:
            # Se áudio muito curto, pad com zeros
            target_width = window_size
            pad_width = target_width - mel_spec.shape[1]
            if pad_width > 0:
                mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
            n_frames = 1
        
        frames = np.zeros((n_frames, mel_spec.shape[0], window_size))
        
        for t in range(n_frames):
            start = t * gap
            end = start + window_size
            frames[t, :, :] = mel_spec[:, start:end]
        return frames

    def preprocess(self, audio_data):
        audio_np = np.array(audio_data, dtype=np.float32)
        mel_spect = self._mel_spectrogram(audio_np, **self._mel_spec_params)
        mel_frames = self._frame_mel_spectrogram(mel_spect, **self._frame_mel_params)

        # Ajusta para 5 frames
        if mel_frames.shape[0] < 5:
            mel_frames = np.pad(mel_frames, ((0, 5 - mel_frames.shape[0]), (0, 0), (0, 0)))
        elif mel_frames.shape[0] > 5:
            mel_frames = mel_frames[:5]

        # Reshape para (1, 5, 128, 128, 1)
        mel_frames = mel_frames.reshape(1, 5, 128, 128, 1)
        return mel_frames
    
class MFCCPreprocessing(PreprocessingStrategy):
    def __init__(self, sample_rate=16000, n_mfcc=40, max_length=3):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_length = max_length

    def _extract_mfcc(self, audio_data, sample_rate=16000, n_mfcc=40, max_length=3):
        y = librosa.util.fix_length(audio_data, size=sample_rate * max_length)
        mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        return mfccs_mean
    
    def preprocess(self, audio_data):
        audio_np = np.array(audio_data, dtype=np.float32)
        mfcc_features = self._extract_mfcc(audio_np, sample_rate=self.sample_rate, n_mfcc=self.n_mfcc)
        mfcc_features = mfcc_features.reshape(1, -1, 1)  # Ajusta para (1, n_mfcc, 1)
        return mfcc_features