import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import gc
gc.collect()
torch.cuda.empty_cache()


def id_retriever(batch_ids, id_list_for_similarity, similarity_matrix, eval_idxs):
    batch_retrived_audio_ids = []
    similarity_matrix[:, eval_idxs] = -1
    for query_id in batch_ids:
        query_idx = id_list_for_similarity.index(query_id)
         
        similarity_scores = similarity_matrix[query_idx]
        # Get the ids of top 1 similar audio
    
        top1_idx = np.argmax(similarity_scores)
        top1_id = id_list_for_similarity[top1_idx]
        batch_retrived_audio_ids.append(top1_id)
    return batch_retrived_audio_ids


# Load simiarlity matrix
similarity_matrix = np.load('./clap_similarity/similarities.npy')
# idx search table
with open("./clap_similarity/id_order_for_similarity_matrix.txt") as f:
    id_list_for_similarity = [line.rstrip('\n') for line in f]

# Initialize model
model = MusicGen.get_pretrained('facebook/musicgen-melody')  # Load the pretrained model
model.set_generation_params(duration=10)  # Generate 10 seconds.

# Read captions from the CSV file
# TODO : create a new csv file with retrieved_audio_filename
musiccaps_df = pd.read_csv('./musiccaps-public.csv')

# Filter the data to use only captions where is_audioset_eval is True
filtered_df = musiccaps_df[musiccaps_df['is_audioset_eval'] == True]
eval_idxs_for_similarity = []
invalid_ids = 0
valid_ids = 0

for idx, row in filtered_df.iterrows():
    if row['ytid'] in id_list_for_similarity:
        eval_idxs_for_similarity.append(id_list_for_similarity.index(row['ytid']))
        valid_ids += 1
    else:
        invalid_ids += 1
print(f"valid_ids: {valid_ids}, invalid_ids: {invalid_ids}")
# only keep the rows that are in the similarity matrix
filtered_df = filtered_df[filtered_df['ytid'].isin(id_list_for_similarity)]
print(f"filtered_df shape: {filtered_df.shape}")

# Extract captions and ids from the filtered DataFrame
captions = filtered_df['caption'].values
ids = filtered_df['ytid'].values

# Define the folder to save audio files
output_folder = './generated_audio_RAG'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Process in batches
batch_size = 8
num_samples = len(captions)
num_batches = num_samples // batch_size
# num_batches = 1 # for testing

# Using tqdm to display a progress bar
with tqdm(total=num_batches, desc="Generating Music", unit="batch") as pbar:
    for batch_idx in range(num_batches):
        torch.cuda.empty_cache()
        gc.collect()
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        batch_captions = captions[start_idx:end_idx]
        batch_ids = ids[start_idx:end_idx]
        batch_retrived_audio_ids = id_retriever(batch_ids, id_list_for_similarity, similarity_matrix, eval_idxs_for_similarity)

        # Generate audio for the current batch
        #batch_wav = model.generate(batch_captions)
        # load batch of audio files
        # melody, sr = torchaudio.load('./assets/bach.mp3')
        for idx, audio_id in enumerate(batch_retrived_audio_ids):
            audio_path = os.path.join('./music_data_new', f"{audio_id}.wav")
            audio, sr = torchaudio.load(audio_path)
            # force into mono
            audio = audio.mean(0).unsqueeze(0)
            new_sr = 48000
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=new_sr)

            # strim audio to 10 seconds
            if audio.shape[1] > new_sr*10:
                audio = audio[:, :new_sr*10]
            else:
                audio = torch.cat((audio, torch.zeros((1, new_sr*10 - audio.shape[1]))), 1)

            audio = audio.unsqueeze(0)
            if idx == 0:
                batch_melodies = audio
            else:
                batch_melodies = torch.cat((batch_melodies, audio), 0)
        
        # generates using the melody from the given audio and the provided descriptions.
        batch_wav = model.generate_with_chroma(batch_captions, batch_melodies, new_sr)

        # Save each generated audio to the specified folder
        for idx, one_wav in enumerate(batch_wav):
            name = batch_ids[idx][1:]  # Remove the leading character if necessary
            file_path = os.path.join(output_folder, f"{name}")  # Save as .wav file in the output folder
            audio_write(file_path, one_wav.cpu(), model.sample_rate, strategy="loudness")
        
        # Update progress bar after each batch is processed
        pbar.update(1)
