import pandas as pd
import os
import pdb 

# File paths
folder = './eval_results'
file1 = 'cosine_similarity_results.csv'
file2 = 'musiccaps-public.csv'
file3 = 'kl_divergence_results.csv'

# Read CSV files
# join the folder path with the file path

df_clap = pd.read_csv(os.path.join(folder, file1))
df_table = pd.read_csv(os.path.join(".", file2))
df_kl = pd.read_csv(os.path.join(folder, file3))

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

## CLAP Analysis
# reduce the df_table['ytid'] only start from second character
df_table['ytid'] = df_table['ytid'].str[1:]
df_clap = df_clap.sort_values('similarity')
filenames_clap = df_clap['filename'].head(10).tolist()
ytid_clap = [filename.split('.')[0] for filename in filenames_clap]
# captions_clap = df_table[df_table['ytid'].isin(ytid_clap)]['caption'].tolist()
captions_clap = [df_table[df_table['ytid'] == ytid]['caption'].tolist()[0] for ytid in ytid_clap]


# Get the last 10 filenames from df_clap
best_filenames_clap = df_clap['filename'].tail(10).tolist()
# Extract ytid from best_filenames_clap
best_ytid_clap = [filename.split('.')[0] for filename in best_filenames_clap]
# Find captions from df_table based on best_ytid_clap
best_captions_clap = [df_table[df_table['ytid'] == ytid]['caption'].tolist()[0] for ytid in best_ytid_clap]



# Print the best captions to a file
with open('analysis.txt', 'w') as f:
    f.write('\n\n')
    f.write('KL Divergence Analysis(Best 10)\n')
    f.write('----------------------\n')
    for i, caption in enumerate(best_captions_kl):
        f.write(f'{i+1}. {caption}\n')
    f.write('\n\n')
    f.write('CLAP Analysis(Best 10)\n')
    f.write('----------------------\n')
    for i, caption in enumerate(best_captions_clap):
        f.write(f'{i+1}. {caption}\n')


# Print the to a file
with open('analysis.txt', 'a') as f:
    f.write('----------------------\n')
    f.write('----------------------\n')
    f.write('\n\n')
    f.write('KL Divergence Analysis(Worst 10)\n')
    f.write('----------------------\n')
    for i, caption in enumerate(captions_kl):
        f.write(f'{i+1}. {caption}\n')
    f.write('\n\n')
    f.write('CLAP Analysis(Worst 10)\n')
    f.write('----------------------\n')
    for i, caption in enumerate(captions_clap):
        f.write(f'{i+1}. {caption}\n')

#iterate over the worst 10 kl's filenames
worst_filenames_kl = df_kl['filename'].head(10).tolist()
worst_filenames_clap = df_clap['filename'].head(10).tolist()
# for loop
#pdb.set_trace()
for filename in worst_filenames_kl:
    # copy audio file to the worst folder
    os.system(f'cp ./musiccaps_gen_eval/{filename[1:]}.wav ./eval_results/audio_samples/worst_kl/generate/')
for filename in best_filenames_kl:
    # copy audio file to the best folder
    os.system(f'cp ./musiccaps_gen_eval/{filename[1:]}.wav ./eval_results/audio_samples/best_kl/generate/')
for filename in worst_filenames_kl:
    # copy audio file to the worst folder
    os.system(f'cp ./musiccaps_eval_strim/{filename} ./eval_results/audio_samples/worst_kl/reference/')
for filename in best_filenames_kl:
    # copy audio file to the best folder
    os.system(f'cp ./musiccaps_eval_strim/{filename} ./eval_results/audio_samples/best_kl/reference/')

for filename in worst_filenames_clap:
    # copy audio file to the worst folder
    filename = filename.split('.')[:-1]
    filename = '.'.join(filename)
    # pdb.set_trace()
    os.system(f'cp ./generated_audio/{filename}.wav ./eval_results/audio_samples/worst_clap/')
for filename in best_filenames_clap:
    # copy audio file to the best folder
    filename = filename.split('.')[:-1]
    filename = '.'.join(filename)
    os.system(f'cp ./generated_audio/{filename}.wav ./eval_results/audio_samples/best_clap/')
    # Generate HTML file
    with open('captions.html', 'w') as f:
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
            f.write(f'<audio controls><source src="./eval_results/audio_samples/best_kl/reference/{best_filenames_kl[i]}" type="audio/wav"></audio>\n')
            f.write(f'<audio controls><source src="./eval_results/audio_samples/best_kl/generate/{best_filenames_kl[i][1:]}.wav" type="audio/wav"></audio>\n')
        f.write('</ul>\n')
        f.write('<h3>Worst 10 Captions</h3>\n')
        f.write('<ul>\n')
        for i, caption in enumerate(captions_kl):
            f.write(f'<li>{caption}</li>\n')
            f.write(f'<audio controls><source src="./eval_results/audio_samples/worst_kl/reference/{worst_filenames_kl[i]}" type="audio/wav"></audio>\n')
            f.write(f'<audio controls><source src="./eval_results/audio_samples/worst_kl/generate/{worst_filenames_kl[i][1:]}.wav" type="audio/wav"></audio>\n')
        f.write('</ul>\n')
        f.write('<h2>CLAP Analysis</h2>\n')
        f.write('<h3>Best 10 Captions</h3>\n')
        f.write('<ul>\n')
        for i, caption in enumerate(best_captions_clap):
            f.write(f'<li>{caption}</li>\n')
            f.write(f'<audio controls><source src="./eval_results/audio_samples/best_clap/{best_filenames_clap[i]}" type="audio/wav"></audio>\n')
        f.write('</ul>\n')
        f.write('<h3>Worst 10 Captions</h3>\n')
        f.write('<ul>\n')
        for i, caption in enumerate(captions_clap):
            f.write(f'<li>{caption}</li>\n')
            f.write(f'<audio controls><source src="./eval_results/audio_samples/worst_clap/{worst_filenames_clap[i]}" type="audio/wav"></audio>\n')
        f.write('</ul>\n')
        f.write('</body>\n')
        f.write('</html>\n')
