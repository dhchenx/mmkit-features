import librosa
import torch
from transformers import Wav2Vec2ForCTC,Wav2Vec2Processor

class TransformerAudioFeatureExtractor:
    def __init__(self,model_or_path="facebook/wav2vec2-base-960h"):
        # Importing Wav2Vec pretrained model
        self.tokenizer = Wav2Vec2Processor.from_pretrained(model_or_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_or_path)

    def extract_features(self,audio_file_path,sample_rate=16000):
        # Loading the audio file
        audio, rate = librosa.load(audio_file_path, sr = sample_rate)
        # Taking an input value
        input_values = self.tokenizer(audio, return_tensors = "pt").input_values
        # Storing logits (non-normalized prediction values)
        logits = self.model(input_values).logits
        return logits

    def predict(self,logits):
        # Storing predicted id's
        prediction = torch.argmax(logits, dim = -1)
        # Passing the prediction to the tokenzer decode to get the transcription
        transcription = self.tokenizer.batch_decode(prediction)[0]
        # print(transcription)
        return transcription
