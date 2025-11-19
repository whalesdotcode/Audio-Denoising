# Denoising of Audio Signals Using Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep learning-based audio denoising using Speech Enhancement Generative Adversarial Network (SEGAN) to remove environmental noise from audio recordings. Developed as part of my undergraduate thesis in Electrical/Electronics Engineering.

## 📊 Results

| Metric | Noisy Audio | Denoised Audio | Improvement |
|--------|-------------|----------------|-------------|
| **SNR (dB)** | 0.86 | 10.08 | **+9.22 dB** |
| **PESQ** | 1.86 | 2.24 | **+0.38** |
| **STOI** | 0.52 | 0.64 | **+23%** |

The model successfully removes environmental noise including dog barking, background chatter, and other ambient sounds while preserving audio quality.

## 🏗️ Architecture

**SEGAN** - Speech Enhancement GAN with encoder-decoder generator and discriminator networks.

**Generator (Denoiser):**
- Input: Noisy audio waveform (80,000 samples @ 16kHz = 5 seconds)
- 5 Conv1D encoder layers (16→32→64→128→256 filters)
- 5 Conv1DTranspose decoder layers (256→128→64→32→16→1)
- Output: Denoised audio waveform

**Training:**
- Framework: TensorFlow 2.x / Keras
- Loss: Adversarial + L1 (weight: 100)
- Optimizer: Adam (lr: 1e-4)
- Epochs: 100, Batch size: 16

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/whalesdotcode/Audio-Denoising.git
cd audio-denoising
pip install -r requirements.txt
```

### Prepare Dataset
1. Place clean audio in `data/clean_audio_folder/`
2. Place noise samples in `data/noise_folder/`
3. Run preprocessing:
```bash
python data_preprocessing.py
```

### Train Model
```bash
jupyter notebook main.ipynb
```

### Denoise Audio
```python
import numpy as np
from tensorflow import keras
import librosa
import soundfile as sf

model = keras.models.load_model('models/audio_denoiser_model.h5')
noisy_audio, sr = librosa.load('noisy.wav', sr=16000)
noisy_audio = np.pad(noisy_audio, (0, max(0, 80000-len(noisy_audio))))[:80000]
denoised = model.predict(noisy_audio.reshape(1, -1, 1)).squeeze()
sf.write('denoised.wav', denoised, sr)
```

## 📁 Project Structure

```
audio-denoising-segan/
├── data/                       # Audio datasets
├── notebooks/main.ipynb        # Training & evaluation
├── data_preprocessing.py       # Data preparation
├── models/                     # Saved models
├── results/                    # Sample outputs
└── requirements.txt
```

## 📈 Evaluation Metrics

- **SNR**: Signal-to-Noise Ratio improvement (9.22 dB average)
- **PESQ**: Perceptual speech quality (1.86 → 2.24)
- **STOI**: Speech intelligibility (0.52 → 0.64)

## 👤 Author

**Olawale Olajumoke**
- B.Eng in Electrical/Electronics Engineering 

