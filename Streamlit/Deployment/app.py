import streamlit as st
import pandas as pd
import seaborn as sns
import whisper 
import numpy as np
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import streamlit as st
import soundfile as sf
from PIL import Image
from pathlib import Path
from tensorflow import keras
import joblib
import io
import wave
import webbrowser
from audio_recorder_streamlit import audio_recorder
import whisper
import flair
from flair.models import TextClassifier
from flair.data import Sentence


logo = 'images/nerdslogo.png'
# st.image(image, caption='Your Logo')

st.set_page_config(
    page_title="Audio Emotion Detection",
    page_icon=logo,
)

# Text Analysis using Whisper

def whisper_totext(audio_file):
    y, sr = librosa.load(io.BytesIO(audio_file))
    model = whisper.load_model('base')
    result = model.transcribe(y, fp16=False)
    return result['text']

def whisperText(audiofile):
    transcript = whisper_totext(audiofile)
    classifier = TextClassifier.load('en-sentiment')
    data = Sentence(transcript)
    classifier.predict(data)
    sentiment = data.labels[0].value
    score = data.labels[0].score
    print(f"Voice to Text Data: {transcript}")
    print(f"Sentiment: {sentiment}")
    print(f"Score: {score}")
    return sentiment, score, transcript

# Functions to get features
   
def feature_methods(data,sample_rate):
    # Zero Crossing
    feature = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    feature=np.hstack((feature, zcr))

    # Chroma
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    feature = np.hstack((feature, chroma_stft))

    # MFCC Base
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    feature = np.hstack((feature, mfcc))

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    feature = np.hstack((feature, rms))

    # Mel Spectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    feature = np.hstack((feature, mel)) 

    #Spectral Contrast
    spectral=np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
    feature= np.hstack((feature,spectral))
    
    return feature
    
def get_features_prediction(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    result = feature_methods(data,sample_rate)
    feature = np.array(result)
    return feature

def CNN(file):
    audio_data = get_features_prediction(io.BytesIO(file))
    meanfile = np.load('models/mean.npy')
    stdfile = np.load('models/std.npy')
    audio_data = audio_data-meanfile
    audio_data = audio_data/stdfile
    model = keras.models.load_model('models/model1_new_architecture.h5')
    model.load_weights('models/best_weights_model_new.h5')
    Xtrain = np.expand_dims(audio_data, axis=-2)
    prediction = model.predict(Xtrain)
    #prediction2 = model2.predict(Xtrain)
    encoder = joblib.load('models/encoder_new.joblib')

    # Decode the encoded array back to the original values
    decoded_array = encoder.inverse_transform(prediction)
    #decoded_array2 = encoder.inverse_transform(prediction2)
    return decoded_array[0][0]

def RNN(file):
    audio_data = get_features_prediction(io.BytesIO(file))
    meanfile = np.load('models/mean.npy')
    stdfile = np.load('models/std.npy')
    audio_data = audio_data-meanfile
    audio_data = audio_data/stdfile
    rnn_model = keras.models.load_model('models/RNN_model.h5')
    rnn_model.load_weights('models/rnn_initial_weights.h5')
    Xtrain = np.expand_dims(audio_data, axis=-2)
    prediction = rnn_model.predict(Xtrain)
    encoder = joblib.load('models/RNNencoder.joblib')

    emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Surprising"]
    df = pd.DataFrame(columns=["Emotion", "Percentages"])

    for i in range(len(Xtrain)):
        percentages = [round(100*prediction[i][j], 2) for j in range(len(emotions))]
        df = df.append(pd.DataFrame({"Emotion": emotions, "Percentages": percentages}))
    return df

    # Decode the encoded array back to the original values
    # decoded_array = encoder.inverse_transform(prediction)
    # return decoded_array[0][0]
def MLP(file):
    audio_data = get_features_prediction(io.BytesIO(file))
    meanfile = np.load('models/mlp_mean.npy')
    stdfile = np.load('models/mlp_std.npy')
    audio_data = audio_data-meanfile
    audio_data = audio_data/stdfile
    Xtrain = np.expand_dims(audio_data, axis=-2)
    mlp_model = keras.models.load_model('models/mlp_model.h5')
    prediction = mlp_model.predict(Xtrain)
    encoder = joblib.load('models/mlp_encoder.joblib')
    decoded_mlp = encoder.inverse_transform(prediction)
    return(decoded_mlp[0][0])

# Functions for graphing features
def mfcc_graph(file):
    y, sr = librosa.load(io.BytesIO(file))

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Plot MFCCs
    sns.set()
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC Visualization')
    plt.tight_layout()
    st.pyplot()

def spec_con_graph(file):
    y, sr = librosa.load(io.BytesIO(file))

    # Compute spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Plot spectral contrast
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(contrast, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.title('Spectral Contrast Visualization')
    plt.tight_layout()
    st.pyplot()

def melspec_graph(file):    
    y, sr = librosa.load(io.BytesIO(file))

    # Compute Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

    # Convert power to dB
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot Mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram Visualization')
    plt.tight_layout()
    st.pyplot()

def chroma_graph(file):
    y, sr = librosa.load(io.BytesIO(file))
    # Compute chroma feature
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Plot chroma feature
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.colorbar()
    plt.title('Chroma Feature Visualization')
    plt.tight_layout()
    st.pyplot()

# Use the audio recorder widget to record audio

def audio_waveform(file):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if (file == None): return
    y, sr = librosa.load(io.BytesIO(file))
    fig, ax = plt.subplots()
    t = np.arange(0, len(y)) / sr
    sns.set(style='ticks', rc={'axes.facecolor': 'none', 'figure.facecolor': 'none'})
    sns.set_style({'axes.spines.bottom': False,'axes.spines.top': False, 'axes.spines.left': False, 'axes.spines.right': False})
    sns.despine()
    plt.figure(figsize=(10, 4))
    plt.xticks([])
    plt.yticks([])
    plt.plot(t, y, color='red')
    plt.xlabel('')
    plt.ylabel('')

    st.pyplot()

def audio_viz(file):
     # Create two columns
    col1, col2 = st.columns(2)
    
    # Display first two graphs in first column
    with col1:
        mfcc_graph(file)
        melspec_graph(file)
    
    # Display last two graphs in second column
    with col2:
        spec_con_graph(file)
        chroma_graph(file)

def emotion_gif(emotion):
    if emotion == 'Happy':
        st.image('images/happy.gif')
    elif emotion == 'Sad':
        st.image('images/sad.gif')
    elif emotion == 'Fear':
        st.image('images/fear.gif')
    elif emotion == 'Neutral':
        st.image('images/neutral.gif')
    elif emotion == 'Angry':
        st.image('images/angry.gif')
    elif emotion == 'Disgust':
        st.image('images/disgust.gif')
    elif emotion == 'Surprise':
        st.image('images/surprise.gif')
    else:
        st.image('images/sad.gif')

def dataset():
    st.header('Dataset')
    st.write('Below is a summary of the CREMA-D Audio Dataset ')

    with st.container():
        st.subheader("Introduction")
        st.write(
            "The CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset) is a publicly available dataset that contains audio and video recordings of 91 professional actors performing scripted and improvised sentences with different emotional expressions." 
            "The dataset was created by researchers at the University of Southern California (USC) and is intended to be used for research and development purposes in the field of affective computing and emotion recognition."
            )
        st.subheader("Audio Recordings")
        st.write(
            "The dataset includes over 7,000 audio recordings, each lasting approximately 5 seconds, with actors speaking from a selection of 12 sentences that convey different emotions such as anger, happiness, sadness, fear, disgust, neutral expressions, and four different emotion levels such as low, medium, high and unspecified. The audio files are recorded in high-quality WAV format with a sampling rate of 44.1 kHz and 16-bit resolution."
            )

def mlalgorithm():
    st.header('Machine Learning Algorithms')
    st.subheader('Convolutional Neural Networks (CNNs)')
    st.write("CNNs are commonly used in speech processing tasks, including emotion recognition. "
             "By converting the audio signal into a spectrogram, CNNs can extract relevant features and classify emotions with high accuracy. "
             "Below is our correlation matrix that shows the how many times our model projected the right emotions.")
    
    col1, col2, col3 = st.columns([1,4,1])
    with col2:
        st.image("images/confusion_matrix_cnn.png")

    st.subheader('Recurrent Neural Networks (RNNs)')
    st.write("RNNs are designed to handle sequential data such as speech signals, making them a good choice for emotion recognition. "
             "They can capture the temporal dependencies in the audio signal and can be used to learn features from the audio. "
             "Below is our correlation matrix that shows the how many times our model projected the right emotions. ")
    
    col1, col2, col3 = st.columns([1,4,1])
    with col2:
        st.image("images/confusion_matrix_rnn.png")

def record():
    audio_bytes = None
    audio_data = None
    model_type = 0
    option = st.selectbox(
    'Select an option:',
    ('Record Audio', 'Upload Audio')
    )
    option2 = st.selectbox(
    'Select Model:',
    ('Convolutional Neural Network (CNN)', 'Recurrent Neural Network (RNN)', 'Multilayer Perceptron (MLP)')
    )
    if option2 == 'Convolutional Neural Network (CNN)':
        st.write(0)
        model_type=0
    elif option2 == 'Recurrent Neural Network (RNN)':
        st.write(1)
        model_type = 1
    elif option2 == 'Multilayer Perceptron (MLP)':
        st.write(2)
        model_type = 2

    if option == 'Record Audio':
        
        audio_data = audio_recorder(energy_threshold=(-1.0, 1.0), pause_threshold=3.0,
                                    text="",
                                    recording_color="#F42727",
                                    neutral_color="#F5F5F5",
                                    icon_name="microphone",
                                    icon_size="4x",
            )
        
        if audio_data:
            st.audio(audio_data, format="audio/wav")
            audio_waveform(audio_data)
    
    elif option == 'Upload Audio':       
        
        uploaded_file = st.file_uploader("Choose a file", type=["wav", "mp3"])

        if uploaded_file is not None:
            audio_data = uploaded_file.read()
            st.audio(audio_data, format="audio/wav")
            audio_waveform(audio_data)

    if st.button('Graph Visualization'):
        audio_viz(audio_data)

        

    if st.button('Detect Emotion'):
        with st.spinner('Processing audio data...'):
            text_analysis = whisperText(audio_data)
            textspeech = ("Text-to-Speech: " 
                        + '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;❝'
                        +text_analysis[2]+ '❞')
        st.success('Done!')
        st.write(textspeech)
        sentiment = text_analysis[0]

        if sentiment == 'POSITIVE':
            st.write("Text-Sentiment:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Positive 😃 ")
        else:
            st.write("Text-Sentiment:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Negative 😡 ")

        if model_type == 0:
            st.write(model_type)
            with st.spinner('Processing audio data...'):
                emotion = CNN(audio_data)
                view_emotion = "Audio Emotion: "+emotion
            st.success('Done!')
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.subheader(view_emotion)
                emotion_gif(emotion)
        elif model_type == 1:
            st.write(model_type)
            with st.spinner('Processing audio data...'):
                emotion = RNN(audio_data)
                max_row = emotion.loc[emotion["Percentages"].idxmax()]
                max_emotion = max_row["Emotion"]
                view_emotion = "Audio Emotion: "+max_emotion
            st.success('Done!')    
            st.subheader(view_emotion)
            emotion["Percentages"] = emotion["Percentages"].apply(lambda x: str(x) + "%")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(emotion)
            with col2:
                emotion_gif(max_emotion)
        elif model_type == 2:
            st.write(model_type)
            with st.spinner('Processing audio data...'):
                emotion = MLP(audio_data)
                view_emotion = "Audio Emotion: "+emotion
            st.success('Done!')
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.subheader(view_emotion)
                emotion_gif(emotion)
            

    
            



def home():
    signal = 'images/signal.png'
    st.image(signal)    
    st.title('Voice Emotion Detection')
    # st.text('This web app uses ML Algorithms to detect realtime emotion of your speech')

    # Add text in a box
    with st.container():
        st.subheader("Introduction")
        st.write(
            "Emotion detection has become a popular research topic in the field of artificial intelligence and machine learning due to its potential applications in various areas, including psychology, marketing, and healthcare. Thus, our team has decided to attempt to tackle this area using modeling techniques on a crowdsourced voice dataset."
            )
        st.write(
            "The goal of this project is to apply the modelling techniques to help classify emotion based on voice data. Multiple models will be experimented with to accomplish this goal, which can be found below in “Model Algorithms”. For the time being, the project will focus on 5 emotions: anger, fear, happiness, sadness and disgust. If time permits our team plans on detecting other emotions."
            )
     
    github_url = 'https://github.com/prempatell/Audio-Emotion-Detection'

    # Center the button
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            margin: 0 auto;
            display: block;
            text-align: center;
            font-size: 40px;
            padding: 14px 24px;
        }
        </style>
        """,
        unsafe_allow_html=True
)  
    if st.button('GitHub'):
        webbrowser.open_new_tab(github_url)


st.sidebar.image(logo, caption='DATA NERDS', use_column_width=True)

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



    


