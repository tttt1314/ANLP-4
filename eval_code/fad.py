from frechet_audio_distance import FrechetAudioDistance
import glob
import os
# to use `vggish`
frechet = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=False
)

# # files in twos folers should match
# files_in_folder1 = os.listdir("../musiccaps_eval_strim")  # list of files in the first folder
# files_in_folder2 = os.listdir("../generated_audio_RAG")  # list of files in the second folder

# # Check if the number of files in both folders match
# if len(files_in_folder1) != len(files_in_folder2):
#     print("Error: Number of files in the two folders do not match")
#     exit()

# Rest of the code...

fad_score = frechet.score(
    "../musiccaps_eval_strim", # path to background set
    "../generated_audio_RAG", # path to eval set

    dtype="float32"
)
print("--FAD---")
# write the score to a file
with open("FAD_RAG_retrieveEvalOnly.txt", "w") as f:
    f.write(str(fad_score))
