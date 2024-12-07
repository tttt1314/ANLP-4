import os
import pandas as pd
import numpy as np
import librosa
import torch
import laion_clap
from tqdm import tqdm
import pdb

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)
# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the table with captions and modify the 'ytid' column
caption_table = pd.read_csv('../musiccaps-public.csv')
caption_table['ytid'] = caption_table['ytid'].apply(lambda x: x[1:] if isinstance(x, str) else x)
caption_table.set_index('ytid', inplace=True)

# Initialize the CLAP model and move it to the GPU if available
model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()  # Download the default pretrained checkpoint.
model = model.to(device)

# Path to audio folder
audio_folder_noRAG = '../generated_audio'
audio_folder = '../generated_audio_RAG'
output_file = 'cosine_similarity_results_noRAG.csv'

# Batch size
batch_size = 8

# Prepare audio filenames
audio_files_RAG = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
audio_files_noRAG = [f[:-4] for f in os.listdir(audio_folder_noRAG) if f.endswith('.wav')]

# calulate to intersection of the two lists
audio_files = list(set(audio_files_RAG).intersection(audio_files_noRAG))
print(len(audio_files_noRAG), len(audio_files_RAG), len(audio_files))
pdb.set_trace()
#audio_files = audio_files_RAG

EVAL_RAG = False

similarities_list = []
# Open the output file for writing
with open(output_file, 'w') as f:
    # Process audio files in batches
    for i in tqdm(range(0, len(audio_files), batch_size), desc="Processing Audio Batches"):
        batch_files = audio_files[i:i + batch_size]
        audio_data_list = []
        captions_list = []
        valid_filenames = []

        # Collect audio data and captions for the batch
        for filename in batch_files:
            ytid = filename.split('.')[0]

            if ytid in caption_table.index:
                caption = caption_table.loc[ytid, 'caption']
                captions_list.append(caption)
                valid_filenames.append(filename)

                # Load and preprocess audio
                if EVAL_RAG:
                    audio_path = os.path.join(audio_folder, filename)
                else:
                    audio_path = os.path.join(audio_folder_noRAG, filename + ".wav") # load the generated audio without RAG
                audio_data = librosa.load(audio_path, sr=48000)[0]
                audio_data = audio_data.reshape(1, -1)  # Make it (1, T)
                #audio_data = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()
                audio_data_list.append(audio_data)
            else:
                print(f"Caption not found for {ytid}, skipping...")

        if not audio_data_list or not captions_list:
            continue  # Skip if no valid data in the batch

        # Convert to tensors and move to GPU
        audio_data_batch = np.vstack(audio_data_list)
        audio_data_batch_tensor = torch.tensor(audio_data_batch, dtype=torch.float32).to(device)

        # Get text embeddings in batch
        text_embeddings_batch = model.get_text_embedding(captions_list)
        text_embeddings_batch_tensor = torch.tensor(text_embeddings_batch, dtype=torch.float32).to(device)

        # Get audio embeddings in batch
        audio_embeddings_batch = model.get_audio_embedding_from_data(x=audio_data_batch)
        audio_embeddings_batch = torch.from_numpy(audio_embeddings_batch).to(device)

        # Compute cosine similarities
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        similarities = cos(audio_embeddings_batch, text_embeddings_batch_tensor)
        similarities = similarities.cpu().numpy()

        # append the results to the list
        similarities_list.extend(list(zip(valid_filenames, similarities)))

# write to a csv file with title
with open(output_file, 'w') as f:
    f.write("filename,similarity\n")
    for filename, similarity in similarities_list:
        f.write(f"{filename},{similarity}\n")


print(f"Results written to {output_file}")
