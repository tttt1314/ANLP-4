import os
import shutil
import glob
import librosa
filenames = os.listdir('./generated_audio')
ytid_col = [filename.split('.')[0] for filename in filenames]
print(len(ytid_col))
# Iterate over each ytid and copy the corresponding file to a folder
count = 0
for ytid in ytid_col:
    # Copy the file with the same ytid to the destination folder
    
    if len(glob.glob(f'./music_data/*{ytid}.wav')) != 0:
        print("Found file {}".format(ytid))
        assert len(glob.glob(f'./music_data/*{ytid}.wav')) == 1
        # for file in glob.glob(f'./music_data/*{ytid}.wav'):
        #     shutil.copy(file, './musiccaps_eval/')
        for file in glob.glob(f'./generated_audio/{ytid}*.wav'):

            shutil.copy(file, './musiccaps_gen_eval/')
    else:
        print(ytid)
        count += 1
    #shutil.copy(f'./music_data/*{ytid}.wav', './musiccaps_eval')
print(count)