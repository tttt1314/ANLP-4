import pandas as pd
import os
import pdb 
import numpy as np
import glob

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

def get_full_id(ytid_list, table):
    full_id_list = []
    for ytid in ytid_list:
        # if ytid matches part of the full ytid
        full_id = table[table['ytid'].str.contains(ytid)]['ytid'].values[0]
        full_id_list.append(full_id)

    return full_id_list

def copy_files(src_folder, filenames, dst_folder, remove_first_char=False):
    for filename in filenames:
        if remove_first_char:
            filename = filename[1:]
        src = os.path.join(src_folder, "*"+filename)
        src_path = glob.glob(src)[0]
        os.system(f'cp "{src_path}" "{dst_folder}"')

# File paths
folder = './eval_results'
file1 = 'clap_scores_RAG_retrieveEvalOnly.csv'
file2 = 'musiccaps-public.csv'
file3 = 'kls_RAG_RetrieveEvalOnly.csv'

# R
# join the folder path with the file path

df_clap = pd.read_csv(os.path.join(folder, file1))
df_table = pd.read_csv(os.path.join(".", file2))
df_kl = pd.read_csv(os.path.join(folder, file3))

## Setup for retrieval
# Load simiarlity matrix
similarity_matrix = np.load('./clap_similarity/similarities.npy')
# idx search table
with open("./clap_similarity/id_order_for_similarity_matrix.txt") as f:
    id_list_for_similarity = [line.rstrip('\n') for line in f]

musiccaps_df = pd.read_csv('./musiccaps-public.csv')
filtered_df = musiccaps_df[musiccaps_df['is_audioset_eval'] == True]
eval_idxs_for_similarity = []

for idx, row in filtered_df.iterrows():
    if row['ytid'] in id_list_for_similarity:
        eval_idxs_for_similarity.append(id_list_for_similarity.index(row['ytid']))
# only keep the rows that are in the similarity matrix
filtered_df = filtered_df[filtered_df['ytid'].isin(id_list_for_similarity)]
print(f"filtered_df shape: {filtered_df.shape}")




## KL Analysis
# Get the first 10 filenames from df_kl
filenames_kl = df_kl['filename'].head(10).tolist()

# Extract ytid from filenames
ytid_kl = [filename.split('.')[0] for filename in filenames_kl]
# Find captions from df_table based on ytid_kl
captions_kl = [df_table[df_table['ytid'] == ytid]['caption'].tolist()[0] for ytid in ytid_kl]

# Get the last 10 filenames from df_kl
best_filenames_kl = df_kl['filename'].tail(10).tolist()
# Extract ytid from best_filenames_kl
best_ytid_kl = [filename.split('.')[0] for filename in best_filenames_kl]
# Find captions from df_table based on best_ytid_kl
# best_captions_kl = df_table[df_table['ytid'].isin(best_ytid_kl)]['caption'].tolist()
best_captions_kl = [df_table[df_table['ytid'] == ytid]['caption'].tolist()[0] for ytid in best_ytid_kl]

worst_ytid_kl_retrieved = id_retriever(ytid_kl, id_list_for_similarity, similarity_matrix, eval_idxs_for_similarity)
worst_captions_kl_retrieved = [df_table[df_table['ytid'] == ytid]['caption'].tolist()[0] for ytid in worst_ytid_kl_retrieved]
worst_filenames_kl_retrieved = [ytid + '.wav' for ytid in worst_ytid_kl_retrieved]
best_ytid_kl_retrieved = id_retriever(best_ytid_kl, id_list_for_similarity, similarity_matrix, eval_idxs_for_similarity)
best_captions_kl_retrieved = [df_table[df_table['ytid'] == ytid]['caption'].tolist()[0] for ytid in best_ytid_kl_retrieved]
best_filenames_kl_retrieved = [ytid + '.wav' for ytid in best_ytid_kl_retrieved]


## CLAP Analysis
# reduce the df_table['ytid'] only start from second character
df_clap = df_clap.sort_values('similarity')
filenames_clap = df_clap['filename'].head(10).tolist()
ytid_clap = [filename.split('.')[0] for filename in filenames_clap]
ytid_clap = get_full_id(ytid_clap, df_table)

# df_table['ytid'] = df_table['ytid'].str[1:]

# captions_clap = df_table[df_table['ytid'].isin(ytid_clap)]['caption'].tolist()
captions_clap = [df_table[df_table['ytid'] == ytid]['caption'].tolist()[0] for ytid in ytid_clap]


# Get the last 10 filenames from df_clap
best_filenames_clap = df_clap['filename'].tail(10).tolist()
# Extract ytid from best_filenames_clap
best_ytid_clap = [filename.split('.')[0] for filename in best_filenames_clap]
best_ytid_clap = get_full_id(best_ytid_clap, df_table)
# Find captions from df_table based on best_ytid_clap
best_captions_clap = [df_table[df_table['ytid'] == ytid]['caption'].tolist()[0] for ytid in best_ytid_clap]

worst_ytid_clap_retrieved = id_retriever(ytid_clap, id_list_for_similarity, similarity_matrix, eval_idxs_for_similarity)
worst_captions_clap_retrieved = [df_table[df_table['ytid'] == ytid]['caption'].tolist()[0] for ytid in worst_ytid_clap_retrieved]
worst_filenames_clap_retrieved = [ytid + '.wav' for ytid in worst_ytid_clap_retrieved]
best_ytid_clap_retrieved = id_retriever(best_ytid_clap, id_list_for_similarity, similarity_matrix, eval_idxs_for_similarity)
best_captions_clap_retrieved = [df_table[df_table['ytid'] == ytid]['caption'].tolist()[0] for ytid in best_ytid_clap_retrieved]
best_filenames_clap_retrieved = [ytid + '.wav' for ytid in best_ytid_clap_retrieved]

# pdb.set_trace()



# Print the best captions to a file
with open('analysis_rag2.txt', 'w') as f:
    f.write('\n\n')
    f.write('KL Divergence Analysis(Best 10)\n')
    f.write('----------------------\n')
    for i, caption in enumerate(best_captions_kl):
        # f.write(f'{i+1}. {caption}\n')
        f.write(f'Text prompt: {caption}\n')
        f.write(f'Retrieved: {best_captions_kl_retrieved[i]}\n')
    f.write('\n\n')
    f.write('CLAP Analysis(Best 10)\n')
    f.write('----------------------\n')
    for i, caption in enumerate(best_captions_clap):
        f.write(f'Text prompt: {caption}\n')
        f.write(f'Retrieved: {best_captions_clap_retrieved[i]}\n')


# Print the to a file
with open('analysis_rag2.txt', 'a') as f:
    f.write('----------------------\n')
    f.write('----------------------\n')
    f.write('\n\n')
    f.write('KL Divergence Analysis(Worst 10)\n')
    f.write('----------------------\n')
    for i, caption in enumerate(captions_kl):
        f.write(f'Text prompt: {caption}\n')
        f.write(f'Retrieved: {worst_captions_kl_retrieved[i]}\n')
    f.write('\n\n')
    f.write('CLAP Analysis(Worst 10)\n')
    f.write('----------------------\n')
    for i, caption in enumerate(captions_clap):
        f.write(f'Text prompt: {caption}\n')
        f.write(f'Retrieved: {worst_captions_clap_retrieved[i]}\n')

#iterate over the worst 10 kl's filenames
worst_filenames_kl = filenames_kl
worst_filenames_clap = filenames_clap
# pdb.set_trace()

## Copy audio files to output directory
# Specify folders
gen_folder = './generated_audio_RAG'      # Folder containing generated evaluation WAV files
ref_folder = './musiccaps_eval_strim'    # Folder containing reference audio files
output_dir = './eval_results/audio_samples_rag'  # Base output directory
retrieved_folder = './music_data_new'  # Folder containing retrieved audio files

best_kl_generate_dir = os.path.join(output_dir, 'best_kl', 'generate')
best_kl_reference_dir = os.path.join(output_dir, 'best_kl', 'reference')
best_kl_retrieved_dir = os.path.join(output_dir, 'best_kl', 'retrieved')
worst_kl_generate_dir = os.path.join(output_dir, 'worst_kl', 'generate')
worst_kl_reference_dir = os.path.join(output_dir, 'worst_kl', 'reference')
worst_kl_retrieved_dir = os.path.join(output_dir, 'worst_kl', 'retrieved')


best_clap_dir = os.path.join(output_dir, 'best_clap')
best_clap_retrieved_dir = os.path.join(output_dir, 'best_clap_retrieved')
worst_clap_dir = os.path.join(output_dir, 'worst_clap')
worst_clap_retrieved_dir = os.path.join(output_dir, 'worst_clap_retrieved')

# Create directories
os.makedirs(best_kl_generate_dir, exist_ok=True)
os.makedirs(best_kl_reference_dir, exist_ok=True)
os.makedirs(best_kl_retrieved_dir, exist_ok=True)
os.makedirs(worst_kl_generate_dir, exist_ok=True)
os.makedirs(worst_kl_reference_dir, exist_ok=True)
os.makedirs(worst_clap_retrieved_dir, exist_ok=True)

os.makedirs(best_clap_dir, exist_ok=True)
os.makedirs(best_clap_retrieved_dir, exist_ok=True)
os.makedirs(worst_clap_dir, exist_ok=True)
os.makedirs(worst_clap_retrieved_dir, exist_ok=True)





# Copy worst KL generated files
copy_files(gen_folder, worst_filenames_kl, worst_kl_generate_dir, remove_first_char=True)

# Copy best KL generated files
copy_files(gen_folder, best_filenames_kl, best_kl_generate_dir, remove_first_char=True)

# Copy worst KL reference files
copy_files(ref_folder, worst_filenames_kl, worst_kl_reference_dir)

# Copy best KL reference files
copy_files(ref_folder, best_filenames_kl, best_kl_reference_dir)

# Copy worst KL retrieved files
copy_files(retrieved_folder, worst_filenames_kl_retrieved, worst_kl_retrieved_dir)

# Copy best KL retrieved files
copy_files(retrieved_folder, best_filenames_kl_retrieved, best_kl_retrieved_dir)

# Copy worst CLAP files from generated audio
copy_files(gen_folder, worst_filenames_clap, worst_clap_dir)

# Copy best CLAP files
copy_files(gen_folder, best_filenames_clap, best_clap_dir)

# Copy worst CLAP retrieved files
copy_files(retrieved_folder, worst_filenames_clap_retrieved, worst_clap_retrieved_dir)

# Copy best CLAP retrieved files
copy_files(retrieved_folder, best_filenames_clap_retrieved, best_clap_retrieved_dir)







with open('captions_rag2.html', 'w') as f:
    f.write('<html>\n')
    f.write('<head>\n')
    f.write('<title>Captions and Audio Files</title>\n')
    f.write('</head>\n')
    f.write('<body>\n')
    f.write('<h1>Captions and Audio Files</h1>\n')

    f.write('<h2>KL Divergence Analysis</h2>\n')
    f.write('<h3>Best 10 Captions</h3>\n')
    f.write('<ul>\n')
    for i, caption in enumerate(best_captions_kl):
        f.write(f'<li>{caption}</li>\n')
        f.write(f'<audio controls><source src="{best_kl_reference_dir}/{best_filenames_kl[i]}" type="audio/wav"></audio>\n')
        f.write(f'<audio controls><source src="{best_kl_retrieved_dir}/{best_filenames_kl_retrieved[i]}" type="audio/wav"></audio>\n')
        generated_filename = best_filenames_kl[i][1:]
        f.write(f'<audio controls><source src="{best_kl_generate_dir}/{generated_filename}" type="audio/wav"></audio>\n')
    f.write('</ul>\n')

    f.write('<h3>Worst 10 Captions</h3>\n')
    f.write('<ul>\n')
    for i, caption in enumerate(captions_kl):
        f.write(f'<li>{caption}</li>\n')
        f.write(f'<audio controls><source src="{worst_kl_reference_dir}/{worst_filenames_kl[i]}" type="audio/wav"></audio>\n')
        f.write(f'<audio controls><source src="{worst_kl_retrieved_dir}/{worst_filenames_kl_retrieved[i]}" type="audio/wav"></audio>\n')
        generated_filename = worst_filenames_kl[i][1:]
        f.write(f'<audio controls><source src="{worst_kl_generate_dir}/{generated_filename}" type="audio/wav"></audio>\n')
    f.write('</ul>\n')

    f.write('<h2>CLAP Analysis</h2>\n')
    f.write('<h3>Best 10 Captions</h3>\n')
    f.write('<ul>\n')
    for i, caption in enumerate(best_captions_clap):
        f.write(f'<li>{caption}</li>\n')
        f.write(f'<audio controls><source src="{best_clap_dir}/{best_filenames_clap[i]}" type="audio/wav"></audio>\n')
        f.write(f'<audio controls><source src="{best_clap_retrieved_dir}/{best_filenames_clap_retrieved[i]}" type="audio/wav"></audio>\n')
    f.write('</ul>\n')

    f.write('<h3>Worst 10 Captions</h3>\n')
    f.write('<ul>\n')
    for i, caption in enumerate(captions_clap):
        f.write(f'<li>{caption}</li>\n')
        f.write(f'<audio controls><source src="{worst_clap_dir}/{worst_filenames_clap[i]}" type="audio/wav"></audio>\n')
        f.write(f'<audio controls><source src="{worst_clap_retrieved_dir}/{worst_filenames_clap_retrieved[i]}" type="audio/wav"></audio>\n')
    f.write('</ul>\n')

    f.write('</body>\n')
    f.write('</html>\n')
