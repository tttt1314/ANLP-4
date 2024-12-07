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


# Initialize model
model = MusicGen.get_pretrained('facebook/musicgen-medium')
model.set_generation_params(duration=10)  # Generate 10 seconds.

# Read captions from the CSV file
musiccaps_df = pd.read_csv('./musiccaps-public.csv')

# Filter the data to use only captions where is_audioset_eval is True
filtered_df = musiccaps_df[musiccaps_df['is_audioset_eval'] == True]

# Extract captions and ids from the filtered DataFrame
captions = filtered_df['caption'].values
ids = filtered_df['ytid'].values

# Define the folder to save audio files
output_folder = './generated_audio'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Process in batches
batch_size = 8
num_samples = len(captions)
num_batches = num_samples // batch_size

# Using tqdm to display a progress bar
with tqdm(total=num_batches, desc="Generating Music", unit="batch") as pbar:
    for batch_idx in range(num_batches):
        torch.cuda.empty_cache()
        gc.collect()
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        batch_captions = captions[start_idx:end_idx]
        batch_ids = ids[start_idx:end_idx]

        # Generate audio for the current batch
        batch_wav = model.generate(batch_captions)

        # Save each generated audio to the specified folder
        for idx, one_wav in enumerate(batch_wav):
            name = batch_ids[idx][1:]  # Remove the leading character if necessary
            file_path = os.path.join(output_folder, f"{name}")  # Save as .wav file in the output folder
            audio_write(file_path, one_wav.cpu(), model.sample_rate, strategy="loudness")
        
        # Update progress bar after each batch is processed
        pbar.update(1)
