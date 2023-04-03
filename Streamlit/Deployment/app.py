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
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


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
    sid = SentimentIntensityAnalyzer()
    transcript = whisper_totext(audiofile)
    polarity= sid.polarity_scores(transcript)
    return transcript, polarity

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
    model = keras.models.load_model('models/cnn_model1_new_architecture.h5')
    model.load_weights('models/cnn_best_weights_model_new.h5')
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

def zero_crossing(file):
    y, sr = librosa.load(io.BytesIO(file))

    # Extract zero-crossing rate feature
    zcr = librosa.feature.zero_crossing_rate(y)

    # Visualize zero-crossing rate and waveform
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 2)
    plt.plot(zcr[0])
    plt.title('Zero-crossing rate')
    plt.xlabel('Frame')
    plt.ylabel('ZCR')
    plt.tight_layout()
    st.pyplot()

def rms(file):
    y, sr = librosa.load(io.BytesIO(file))

    # Extract zero-crossing rate feature
    zcr = librosa.feature.zero_crossing_rate(y)

    rms = librosa.feature.rms(y=y)

    # Normalize RMS value to the range [0, 1]
    rms_norm = np.interp(rms, (rms.min(), rms.max()), (0, 1))

    # Visualize waveform and RMS value
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 2)
    plt.plot(rms_norm[0])
    plt.title('Root Mean Square (RMS) Value')
    plt.xlabel('Frame')
    plt.ylabel('RMS Value')
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
        zero_crossing(file)
    
    # Display last two graphs in second column
    with col2:
        spec_con_graph(file)
        chroma_graph(file)
        rms(file)

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
    st.header('Datasets')

    with st.container():
        st.write(
            "The [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) dataset, or the Toronto Emotional Speech Set, is a collection of audio recordings of 200 actors speaking scripted sentences with various emotional expressions, including anger, fear, happiness, sadness, and neutral. The recordings are of high quality and provide consistent labeling of emotions, making it an excellent dataset for training and evaluating emotion detection algorithms."
            )
        st.write(
            "The [RAVDESS](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) dataset, or the Ryerson Audio-Visual Database of Emotional Speech and Song, contains audio recordings of 24 professional actors speaking sentences with different emotions, including neutral, calm, happy, sad, angry, fearful, surprise, and disgust. Additionally, the dataset includes audio recordings of singing with emotions and vocalizations expressing different emotions."
            )
        st.write(
            "The [SAVEE](https://www.kaggle.com/ejlok1/surrey-audiovisual-expressed-emotion-savee) dataset, or the Toronto Emotional Speech Set, is a collection of audio recordings of 200 actors speaking scripted sentences with various emotional expressions, including anger, fear, happiness, sadness, and neutral. The recordings are of high quality and provide consistent labeling of emotions, making it an excellent dataset for training and evaluating emotion detection algorithms."
            )
        st.write(
            "The [Crema-D](https://www.kaggle.com/ejlok1/cremad) dataset, or the Toronto Emotional Speech Set, is a collection of audio recordings of 200 actors speaking scripted sentences with various emotional expressions, including anger, fear, happiness, sadness, and neutral. The recordings are of high quality and provide consistent labeling of emotions, making it an excellent dataset for training and evaluating emotion detection algorithms."
            )
        
        st.image("images/emotion_data.png")
        

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
    
    st.subheader('Multilayer Perceptron (MLP)')
    st.write("Multilayer Perceptron (MLP) is a type of artificial neural network (ANN) that has been used for various applications such as classification and regression problems."
             "One of the advantages of MLP is that it can learn complex non-linear relationships between the input features and the emotional content of the audio signal." 
             "This is important for audio emotion detection because emotions are not always easy to categorize and can be influenced by various factors, such as intonation, pitch, and tempo")

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
                        +text_analysis[0]+ '❞')
        st.success('Done!')
        st.write(textspeech)

        sentiments = {'1':text_analysis[1]}
        for key, value in sentiments['1'].items():
            sentiments['1'][key]= f"{round(value*100, 1)}%"

        neutral = 'Neutral: '+ str(text_analysis[1].get('neu'))
        positive = 'Postivie: '+str(text_analysis[1].get('pos'))
        negative = 'Negative: '+str(text_analysis[1].get('neg'))
        compound = 'Compound: '+str(text_analysis[1].get('compound'))
        spacing = 'Text-Sentiment:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        
        # st.write(spacing+neutral)
        # st.write(spacing+positive)
        # st.write(spacing+negative)
        # st.write(spacing+compound)

    
        st.write(neutral,'😐  |  ', positive, '😃  |  ', negative, '😡  |  ', compound, '🤡')
            

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



    


