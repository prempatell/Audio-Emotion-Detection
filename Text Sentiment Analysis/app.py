import speech_recognition
import whisper  
import flair
from flair.models import TextClassifier
from flair.data import Sentence
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()


def S2T_flair(audio_file):
    rec = speech_recognition.Recognizer()
    audio_file = audio_file

    with speech_recognition.AudioFile(audio_file) as audio_data:
        audio_data = rec.record(audio_data)
        result = rec.recognize_google(audio_data, show_all=True)
        transcript = result['alternative'][0]["transcript"]
    classifier = TextClassifier.load('en-sentiment')
    data = Sentence(transcript)
    classifier.predict(data)
    sentiment = data.labels[0].value
    score = data.labels[0].score
    polarity= sid.polarity_scores(transcript)
    print(f"Voice to Text Data: {transcript}")
    print(f"Sentiment: {sentiment}")
    print(f"Flair Score: {score}")
    print(f"Polarity Score: {polarity}")
    return sentiment, score, transcript

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

audio_file= r"AudioWAV\1001_TIE_DIS_XX.wav"

S2T_flair(audio_file)
whisperText(audio_file)