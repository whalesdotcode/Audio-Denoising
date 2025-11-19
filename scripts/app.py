import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import io
import tensorflow as tf
from scipy.io import wavfile
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from aiortc.contrib.media import MediaRecorder

def make_generator_model():
    inputs = layers.Input(shape=(80000, 1))
    
    # Encoder
    x = layers.Conv1D(16, 32, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1D(32, 32, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1D(64, 32, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1D(128, 32, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1D(256, 32, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    # Decoder
    x = layers.Conv1DTranspose(128, 32, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1DTranspose(64, 32, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1DTranspose(32, 32, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1DTranspose(16, 32, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    outputs = layers.Conv1DTranspose(1, 32, strides=2, padding='same', activation='tanh')(x)
    
    return models.Model(inputs, outputs)

def make_discriminator_model():
    inputs = layers.Input(shape=(80000, 1))
    
    x = layers.Conv1D(16, 32, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1D(32, 32, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1D(64, 32, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1D(128, 32, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv1D(256, 32, strides=2, padding='same')(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    
    return models.Model(inputs, outputs)




@tf.keras.utils.register_keras_serializable()
class SEGAN(keras.Model):
    def __init__(self):
        super(SEGAN, self).__init__()
        self.generator = make_generator_model()
        self.discriminator = make_discriminator_model()
    
    def call(self, inputs):
        return self.generator(inputs)
    
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(SEGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
    
    def train_step(self, data):
        noisy, clean = data
        
        # Train Discriminator
        with tf.GradientTape() as tape:
            fake_clean = self.generator(noisy, training=True)
            real_output = self.discriminator(clean, training=True)
            fake_output = self.discriminator(fake_clean, training=True)
            d_loss = self.loss_fn(tf.ones_like(real_output), real_output) + \
                     self.loss_fn(tf.zeros_like(fake_output), fake_output)
        
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Train Generator
        with tf.GradientTape() as tape:
            fake_clean = self.generator(noisy, training=True)
            fake_output = self.discriminator(fake_clean, training=True)
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output) + \
                     100 * tf.reduce_mean(tf.abs(clean - fake_clean))  # L1 loss
        
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        return {"d_loss": d_loss, "g_loss": g_loss}

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


with tf.keras.utils.custom_object_scope({'SEGAN': SEGAN}):
    model = tf.keras.models.load_model('/Users/whales_mac/Desktop/wave_u_net denoiser/new_segan_model.keras')
def preprocess_audio(audio, sr):
    # Ensure audio is mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample to 16kHz if necessary
    if sr != 16000:
        audio = librosa.resample(audio, sr, 16000)
    
    # Pad or truncate to 5 seconds (80000 samples)
    if len(audio) < 80000:
        audio = np.pad(audio, (0, 80000 - len(audio)))
    else:
        audio = audio[:80000]
    
    return audio

def denoise_audio(audio):
    # Reshape audio for model input
    audio_input = audio.reshape(1, 80000, 1)
    
    # Get model prediction
    denoised = model.predict(audio_input)
    
    return denoised.reshape(-1)

def plot_waveforms(noisy, denoised):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(noisy)
    ax1.set_title('Original (Noisy) Audio Waveform')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    
    ax2.plot(denoised)
    ax2.set_title('Denoised Audio Waveform')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Amplitude')
    
    plt.tight_layout()
    return fig

st.set_page_config(layout="wide")

st.title('Audio Denoising App')

left_column, right_column = st.columns(2)

with left_column:
    st.header("Audio Input and Playback")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a noisy audio file", type=['wav', 'mp3'])

    if uploaded_file is not None:
        # Load the audio
        audio, sr = librosa.load(uploaded_file, sr=None)
        
        # Preprocess the audio
        processed_audio = preprocess_audio(audio, sr)
        
        # Denoise the audio
        denoised_audio = denoise_audio(processed_audio)
        
        # Play original and denoised audio
        st.subheader("Original (Noisy) Audio")
        st.audio(uploaded_file, format='audio/wav')
        
        st.subheader("Denoised Audio")
        denoised_buffer = io.BytesIO()
        sf.write(denoised_buffer, denoised_audio, 16000, format='WAV')
        st.audio(denoised_buffer, format='audio/wav')

    st.header("Or Record Audio")
    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        audio_receiver_size=1024,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": False, "audio": True},
    )

    if webrtc_ctx.audio_receiver:
        if st.button("Process recorded audio"):
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                audio_data = []
                for frame in audio_frames:
                    audio_data.extend(frame.to_ndarray().flatten())
                
                audio_array = np.array(audio_data)

                # Resample to 16kHz if necessary
                if frame.sample_rate != 16000:
                    audio_array = librosa.resample(audio_array, frame.sample_rate, 16000)

                # Preprocess the audio
                processed_audio = preprocess_audio(audio_array, 16000)

                # Denoise the audio
                denoised_audio = denoise_audio(processed_audio)

                # Play original and denoised audio
                st.subheader("Original (Noisy) Recorded Audio")
                st.audio(processed_audio, sample_rate=16000)

                st.subheader("Denoised Recorded Audio")
                st.audio(denoised_audio, sample_rate=16000)

            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")

with right_column:
    st.header("Waveform Visualization")
    
    if 'processed_audio' in locals() and 'denoised_audio' in locals():
        fig = plot_waveforms(processed_audio, denoised_audio)
        st.pyplot(fig)
    else:
        st.write("Upload or record audio to see waveforms.")