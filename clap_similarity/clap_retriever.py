import os
import pandas as pd
import numpy as np
import librosa
import torch
import laion_clap
from tqdm import tqdm

caption_table = pd.read_csv('./musiccaps-public.csv')

# defind a matrix, with text * audio
# so each row means the similarity between a specific text and all audios
# get number of files in a folder
def get_num_files(folder):
    return len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])

audio_embeddings_folder = "./clap_embeddings/audio"
text_embeddings_folder = "./clap_embeddings/text"
num_audio_files = get_num_files(audio_embeddings_folder)
num_text_files = get_num_files(text_embeddings_folder)
print(num_audio_files, num_text_files)
assert num_audio_files == num_text_files

similarities = np.zeros((num_text_files, num_audio_files))

# get filenames of text embeddings
text_filenames = os.listdir(text_embeddings_folder)
# text_filenames = text_filenames.sort()
print(text_filenames)

# save filnames order in a file
with open("id_order_for_similarity_matrix.txt", "w") as f:
    for filename in text_filenames:
        f.write(filename.split(".")[0] + "\n")

text_embeddings = []
audio_embeddings = []
for filename in text_filenames:
    # load text embeddings, numpy
    text_embedding = np.load(os.path.join(text_embeddings_folder, filename))
    text_embeddings.append(torch.tensor(text_embedding))
    audio_embedding = np.load(os.path.join(audio_embeddings_folder, filename))
    audio_embeddings.append(torch.tensor(audio_embedding))

text_embeddings = torch.stack(text_embeddings)
audio_embeddings = torch.stack(audio_embeddings)
print(text_embeddings.shape)

# calculate similarities
for i in tqdm(range(num_text_files)):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # broadcast text_embedding[i] to the same shape as audio_embeddings
    # then calculate cosine similarity
    similarities[i] = cos(text_embeddings[i].expand_as(audio_embeddings), audio_embeddings).detach().numpy()

# save similarities
np.save("similarities.npy", similarities)

