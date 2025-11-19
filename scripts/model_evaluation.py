import os
import numpy as np
import librosa
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
from pesq import pesq
from scipy.io.wavfile import write as write_wav

from scripts.model import WaveUNet
 # Ensure your WaveUNet model is defined in this path

class AudioDataset(Dataset):
    """Custom dataset for loading preprocessed audio features for evaluation."""
    def __init__(self, noisy_dir, clean_dir=None):
        self.noisy_files = [os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith('.npy')]
        self.clean_files = None
        if clean_dir:
            self.clean_files = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_features = np.load(self.noisy_files[idx])
        noisy_features = torch.tensor(noisy_features, dtype=torch.float32).unsqueeze(0)
        
        if self.clean_files:
            clean_features = np.load(self.clean_files[idx])
            clean_features = torch.tensor(clean_features, dtype=torch.float32).unsqueeze(0)
            return noisy_features, clean_features
        
        return noisy_features

def load_model(model_path, input_channels=1, output_channels=1):
    """Load the trained model."""
    model = WaveUNet(input_channels=input_channels, output_channels=output_channels)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def inverse_mfcc(mfcc, sr):
    """Convert MFCC features back to waveform."""
    return librosa.feature.inverse.mfcc_to_audio(mfcc.T, sr=sr)

def compute_snr(clean_signal, denoised_signal):
    """Compute Signal-to-Noise Ratio (SNR)."""
    noise = clean_signal - denoised_signal
    snr = 10 * np.log10(np.sum(clean_signal ** 2) / np.sum(noise ** 2))
    return snr

def compute_pesq(clean_path, denoised_path, sr):
    """Compute Perceptual Evaluation of Speech Quality (PESQ)."""
    pesq_score = pesq(sr, clean_path, denoised_path, 'wb')
    return pesq_score

def evaluate_model(model, dataloader, sample_rate=16000):
    """Evaluate the model and compute metrics."""
    mse_list, snr_list, pesq_list = [], [], []
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    for idx, batch in enumerate(dataloader):
        if len(batch) == 2:  # Supervised evaluation
            inputs, targets = batch
            with torch.no_grad():
                outputs = model(inputs).squeeze(1).numpy()  # Remove channel dim for MFCC
                
            targets_np = targets.squeeze(1).numpy()
            
            for i, (output, target) in enumerate(zip(outputs, targets_np)):
                mse = mean_squared_error(target, output)
                mse_list.append(mse)

                clean_audio = inverse_mfcc(target, sample_rate)
                denoised_audio = inverse_mfcc(output, sample_rate)
                snr = compute_snr(clean_audio, denoised_audio)
                snr_list.append(snr)

                clean_path = os.path.join(results_dir, f'clean_{idx}_{i}.wav')
                denoised_path = os.path.join(results_dir, f'denoised_{idx}_{i}.wav')
                write_wav(clean_path, sample_rate, (clean_audio * 32767).astype(np.int16))
                write_wav(denoised_path, sample_rate, (denoised_audio * 32767).astype(np.int16))
                pesq_score = compute_pesq(clean_path, denoised_path, sample_rate)
                pesq_list.append(pesq_score)

                # Save output for inspection
                output_path = os.path.join(results_dir, f'denoised_{idx}_{i}_mfcc.npy')
                np.save(output_path, output)

        else:  # Unsupervised evaluation
            inputs = batch
            with torch.no_grad():
                outputs = model(inputs).squeeze(1).numpy()

            for i, output in enumerate(outputs):
                denoised_audio = inverse_mfcc(output, sample_rate)
                output_path = os.path.join(results_dir, f'denoised_{idx}_{i}.wav')
                write_wav(output_path, sample_rate, (denoised_audio * 32767).astype(np.int16))

    avg_mse = np.mean(mse_list) if mse_list else None
    avg_snr = np.mean(snr_list) if snr_list else None
    avg_pesq = np.mean(pesq_list) if pesq_list else None
    print(f'Average MSE: {avg_mse}')
    print(f'Average SNR: {avg_snr}')
    print(f'Average PESQ: {avg_pesq}')
    return avg_mse, avg_snr, avg_pesq

def main():
    test_noisy_dir = 'data/processed/noisy_test'
    test_clean_dir = 'data/processed/clean_test'  # Optional: only if ground truth is available for supervised evaluation
    model_path = 'models/saved_models/wave_unet_mfcc.pth'
    sample_rate = 16000

    # Prepare dataset and dataloader
    test_dataset = AudioDataset(noisy_dir=test_noisy_dir, clean_dir=test_clean_dir)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Load model
    model = load_model(model_path)

    # Evaluate model
    evaluate_model(model, test_loader, sample_rate)

if __name__ == "__main__":
    main()