import os
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, ASTForAudioClassification
import torchaudio
import numpy as np
import pdb 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Load the AST model and feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

model = model.to(device)
#feature_extractor = feature_extractor.to(device)

# Function to load and preprocess audio
def load_audio(file_path, target_sample_rate):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform[0].numpy()  # Convert to 1D numpy array

# Function to calculate KL divergence
def kl_divergence(logits1, logits2):
    return F.kl_div(
        F.log_softmax(logits1, dim=-1),
        F.softmax(logits2, dim=-1),
        reduction='none'
    )

# Paths to original and generated audio folders
original_folder = '../musiccaps_eval_strim'
generated_folder = '../generated_audio_RAG'

# Get sample rate from the model's feature extractor
sampling_rate = feature_extractor.sampling_rate

from tqdm import tqdm
import os
import torch
import numpy as np

# File to store the output
output_file = "kl_RAG.txt"

# Initialize variables
kl_values = []
batch_size = 16  # Adjust this value as needed
audio_batches = []
file_batches = []

# Prepare batches of filenames
filenames_orig = [f[1:] for f in os.listdir(original_folder) if f.endswith('.wav')]
filenames_gen = [f for f in os.listdir(generated_folder) if f.endswith('.wav')]
# take the intersection of the two lists
filenames_temp = list(set(filenames_orig) & set(filenames_gen))
print(len(filenames_orig), len(filenames_gen), len(filenames_temp))
filenames = []
for f in os.listdir(original_folder):
    if f[1:] in filenames_temp:
        filenames.append(f)

assert len(filenames) == len(filenames_temp)

        

# test 160 files
# filenames = filenames[:160]
for i in range(0, len(filenames), batch_size):
    
    batch_filenames = filenames[i:i + batch_size]
    audio_batches.append(batch_filenames)



# Open the file for writing
with open(output_file, "w") as f:
    # Iterate over batches using tqdm for progress tracking
    for batch_filenames in tqdm(audio_batches, desc="Processing Batches"):
        original_audio_batch = []
        generated_audio_batch = []
        
        # Load audio data in batches
        for filename in batch_filenames:
            original_audio = load_audio(os.path.join(original_folder, filename), sampling_rate)
            generated_audio = load_audio(os.path.join(generated_folder, filename[1:]), sampling_rate)
            
            original_audio_batch.append(original_audio)
            generated_audio_batch.append(generated_audio)
        
        # Extract features for the batch
        inputs_original = feature_extractor(original_audio_batch, sampling_rate=sampling_rate, return_tensors="pt", padding=True).to(device)
        inputs_generated = feature_extractor(generated_audio_batch, sampling_rate=sampling_rate, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            logits_original = model(**inputs_original).logits
            logits_generated = model(**inputs_generated).logits

        # Calculate KL divergence (inclueded softmax)
        kl_batch = kl_divergence(logits_original, logits_generated).cpu().numpy()
    
        # Print average KL value for every processed batch
        kl_batch = np.sum(kl_batch, axis=1)
        kl_values.extend([(batch_filenames[i], kl_batch[i]) for i in range(len(batch_filenames))])
        average_kl = np.mean(kl_batch)

        print("current average kl: ", average_kl)
        f.write(f"Processed batch of {len(batch_filenames)} files. Current Batch Average KL divergence: {average_kl:.4f}\n")

    # Print final average KL value
    final_average_kl = np.mean([x[1] for x in kl_values])
    f.write(f"Final Average KL divergence: {final_average_kl:.4f}\n")

    # Sort and report the top 10 highest KL divergence samples
    kl_values.sort(key=lambda x: x[1], reverse=True)
    f.write("Top 10 highest KL divergence samples:\n")
    for filename, kl in kl_values[:10]:
        f.write(f"{filename}: KL divergence = {kl:.4f}\n")
    
    kl_values = np.array(kl_values)

# save kl divergence values to a csv table use pandas
import pandas as pd
df = pd.DataFrame(kl_values, columns=['filename', 'kl_divergence'])
df.to_csv('kl_divergence_results_RAG.csv', index=False)



print(f"KL divergence results have been written to {output_file}")

    