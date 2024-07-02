import os
import torch
import torchaudio
import numpy as np
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2Config
from jiwer import cer 

#Load and Prepare model
model_name = "facebook/wav2vec-base-xlrv"
config = Wav2Vec2Config.from_pretrained(model_name)
config.tokenizer_class = "Wav2Vec2CTCTokenizer"

processor = Wav2Vec2Processor.from_pretrained(model_name, config=config)
model = Wav2Vec2ForCTC.from_pretrained(model_name, config=config)                                                                                                                                                data_path = './data'

# Dictinary to hold datasets
datasets = {
    'train-clean-100': torchaudio.datasets.LIBRISPEECH(data_path, url="train-clean-100", download=True, force_download=True),
    'train-clean-360': torchaudio.datasets.LIBRISPEECH(data_path, url="train-clean-360", download=True, force_download=True),
    'test-clean': torchaudio.datasets.LIBRISPEECH(data_path, url="test-clean", download=True, force_download=True),
    'test-other': torchaudio.datasets.LIBRISPEECH(data_path, url="test-other", download=True, force_download=True)
}  

# Define Speech Recognition function
def speech_to_text(model, processor, waveform, sampling_rate):
    # Ensure waveform is mono and resample if necessary
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Process waveform through the model
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription 
  
# Evaluate the Model
def evaluate_model_on_datasets(model, processor, datasets, num_samples=100):
    results = {}
    for name, dataset in datasets.items():
        total_cer = 0
        num_samples = min(num_samples, len(dataset))
        for i in range(num_samples):
            waveform, sample_rate, transcript, _, _, _ = dataset[i]
            transcription = speech_to_text(model, processor, waveform, sample_rate)
            reference = transcript
            total_cer += cer(reference, transcription)
        average_cer = total_cer / num_samples
        results[name] = average_cer
        print(f"Average CER for {name}: {average_cer:.4f}")
    return results                                                                                                                                                                                                                                        results = evaluate_model_on_datasets(model, processor, datasets)
# Print the results of the evaluation
print(results)
