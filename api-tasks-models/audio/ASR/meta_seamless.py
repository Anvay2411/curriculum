import torch
from transformers import SeamlessM4TModel, AutoProcessor
import librosa

# Load the model and processor
model_id = "facebook/hf-seamless-m4t-large"  # Or another SeamlessM4T model
model = SeamlessM4TModel.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(model_id)

# Load audio file
audio_filepath = "./samples/modi_speech.wav"
audio_input, sr = librosa.load(audio_filepath, sr=16000, mono=True) # Ensure consistent sample rate

# Preprocess the audio
inputs = processor(audio_input, sampling_rate=sr, return_tensors="pt").to(model.device)


# Perform inference
with torch.no_grad():
    generated_ids = model.generate(**inputs)

# Decode the generated IDs to text
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(transcription)