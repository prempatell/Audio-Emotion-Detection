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

def dataset():
    st.header('Dataset')
    st.write('Below is a summary of the CREMA-D Audio Dataset ')

def mlalgorithm():
    st.header('Dataset')
    st.write('Below is a summary of the CREMA-D Audio Dataset ')

def record():
    st.header("1. Record your own voice")
    st.subheader("Audio Recorder")

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

def home():
    st.title('Voice Emotion Detection')
    st.text('This web app uses ML Algorithms to detect realtime emotion of your speech')

st.sidebar.title('Navigation')
options = st.sidebar.radio('Pages', options=['Home','Dataset', 'ML Algorithims' ,'Emotion Detection'])

if options == 'Home':
    home()
elif options == 'Dataset':
    dataset()
elif options == 'ML Algorithims':
    mlalgorithm()
elif options == 'Emotion Detection':
    record()



    


