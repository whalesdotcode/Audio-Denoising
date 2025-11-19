import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm

def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def add_noise_to_clean_audio(clean_audio, noise_audio, snr_db):
    clean_rms = np.sqrt(np.mean(clean_audio**2))
    noise_rms = np.sqrt(np.mean(noise_audio**2))
    snr = 10**(snr_db/20)
    noise_scale = clean_rms / (noise_rms * snr)
    noise_scaled = noise_scale * noise_audio
    return clean_audio + noise_scaled[:len(clean_audio)]

def prepare_dataset(clean_audio_folder, noise_folder, processed_audio, segment_length=5, sr=16000, snr_range=(-5, 15)):
    # Load clean audio files
    clean_audio_files = [f for f in os.listdir(clean_audio_folder) if f.endswith('.wav')]
    clean_audio_files.sort()  # Ensure consistent ordering

    # Load noise samples
    noise_files = [f for f in os.listdir(noise_folder) if f.endswith('.wav')]
    noise_files.sort()  # Ensure consistent ordering

    # Ensure we have at least as many noise samples as clean audio files
    if len(noise_files) < len(clean_audio_files):
        raise ValueError("Not enough noise samples for each clean audio file.")

    # Create output folders
    clean_output_folder = os.path.join(output_folder, 'clean')
    noisy_output_folder = os.path.join(output_folder, 'noisy')
    os.makedirs(clean_output_folder, exist_ok=True)
    os.makedirs(noisy_output_folder, exist_ok=True)

    # Segment length in samples
    segment_samples = segment_length * sr

    file_count = 0

    # Process each clean audio file with its corresponding noise sample
    for clean_file, noise_file in zip(clean_audio_files, noise_files):
        print(f"Processing {clean_file} with {noise_file}")
        
        clean_path = os.path.join(clean_audio_folder, clean_file)
        noise_path = os.path.join(noise_folder, noise_file)
        
        clean_audio = load_audio(clean_path, sr)
        noise = load_audio(noise_path, sr)

        # Generate noisy and clean segments
        for i in tqdm(range(0, len(clean_audio), segment_samples)):
            clean_segment = clean_audio[i:i+segment_samples]
            
            # If the segment is not full length, pad it
            if len(clean_segment) < segment_samples:
                clean_segment = np.pad(clean_segment, (0, segment_samples - len(clean_segment)))
            
            # If noise is shorter than segment, repeat it
            if len(noise) < segment_samples:
                noise_segment = np.tile(noise, (segment_samples // len(noise) + 1))[:segment_samples]
            else:
                start = np.random.randint(0, len(noise) - segment_samples)
                noise_segment = noise[start:start+segment_samples]
            
            # Random SNR
            snr = np.random.uniform(*snr_range)
            
            # Add noise to clean segment
            noisy_segment = add_noise_to_clean_audio(clean_segment, noise_segment, snr)
            
            # Save segments
            clean_path = os.path.join(clean_output_folder, f'clean_{file_count:04d}.wav')
            noisy_path = os.path.join(noisy_output_folder, f'noisy_{file_count:04d}.wav')
            
            sf.write(clean_path, clean_segment, sr)
            sf.write(noisy_path, noisy_segment, sr)
            
            file_count += 1

    print(f"Total number of files created in each folder: {file_count}")

# Usage
clean_audio_folder = 'data/clean_audio_folder'
noise_folder = 'data/noise_folder'
output_folder = 'data/processed_audio'

prepare_dataset(clean_audio_folder, noise_folder, processed_audio)