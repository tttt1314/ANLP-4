import os
import librosa
import soundfile as sf
import shutil
# get all files in the directory
files = os.listdir("./musiccaps_eval/")
# iterate over all files
for file in files:
    # get youtube id
    ytid = file.split(".")[0]
    # load audio file
    y, sr = librosa.load("./musiccaps_eval/"+file, sr=None)
    time = librosa.get_duration(y=y, sr=sr)
    if time > 10:
        print("Time is greater than 10 seconds")
        # only use last 10 seconds
        y = y[-10*sr:]
        path = "./musiccaps_eval_strim/"+ytid+".wav"
        #librosa.output.write_wav(path, y, sr)
        sf.write(path, y, sr, subtype='PCM_24')
        print("Saved audio file:", path)
    else:
        shutil.copyfile("./musiccaps_eval/"+file, "./musiccaps_eval_strim/"+file)
        print("Time is less than or equal to 10 seconds")