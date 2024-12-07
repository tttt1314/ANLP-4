import numpy as np

import matplotlib.pyplot as plt

# Create a random numpy matrix
matrix = np.load('similarities.npy')
# only take 100X100
matrix = matrix[:100, :100]

# create a max mask, 1 when the value is the maximum in the row
max_mask = matrix == matrix.max(axis=1)[:, None]
# Plot the mask
plt.imshow(max_mask, cmap='gray')
plt.show()
# save the mask
plt.imsave('max_mask.png', max_mask, cmap='gray')