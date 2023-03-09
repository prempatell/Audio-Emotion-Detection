import streamlit as st
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import streamlit as st
import soundfile as sf
from pathlib import Path
import io
import wave
from audio_recorder_streamlit import audio_recorder



st.title('Voice Emotion Detection')
st.text('This web app uses ML Algorithms to detect realtime emotion of your speech')


st.header("1. Record your own voice")

# filename = st.text_input("Choose a filename: ")
# audio_bytes = audio_recorder(pause_threshold=2.0, sample_rate=41_000)

# Define a function to visualize the recorded audio




st.subheader("Audio Recorder")


# Use the audio recorder widget to record audio

def audio_waveform(audio):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if (audio == None): return
    data, sr = sf.read(io.BytesIO(audio))
    #y, sr = librosa.load(audio)
    fig, ax = plt.subplots()
    plt.figure(figsize=(14, 5))
    plt.plot(data)
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("Audio waveform")
    st.pyplot()

# Create a Streamlit app

#st.write("Click the button below to start recording")

# Call the audio_recorder() function to record audio
audio_data = audio_recorder(energy_threshold=(-1.0, 1.0), pause_threshold=2.0,
                            text="Record your voice",
                            recording_color="#e8b62c",
                            neutral_color="#6aa36f",
                            icon_name="microphone",
                            icon_size="6x",
    )

if audio_data:
    st.audio(audio_data, format="audio/wav")


# Plot the audio waveform using the plot_audio_waveform function
audio_waveform(audio_data)

    


