import numpy as np

# Specify the path to the .npy file
file_path = 'similarities.npy'

# Load the matrix using np.load()
matrix = np.load(file_path)

# Now you can use the loaded matrix in your code
# For example, you can print the shape of the matrix

# print original data type
print("Matrix dtype:", matrix.dtype)
# calculate the memory usage
memory_usage = matrix.nbytes / 1024 / 1024
print("Memory usage (before):", memory_usage, "MB")
# float64 to float16
matrix = matrix.astype(np.float16)
# calculate the memory usage
memory_usage = matrix.nbytes / 1024 / 1024
print("Memory usage (after):", memory_usage, "MB")
# save it
np.save('similarities_compressed.npy', matrix)