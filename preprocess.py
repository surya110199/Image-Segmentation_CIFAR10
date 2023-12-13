import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib import pyplot as plt
horse_images = np.load(os.path.join('Data', 'horse_images.npy')) 
img = horse_images[1000]
img = img.reshape(-1, 3)

plt.imshow(img)
plt.show()

colours = np.random.randint(0, 256, size=(5, 3))

distances = np.zeros((img.shape[0], colours.shape[0]))

for i in range(img.shape[0]):
    for j in range(colours.shape[0]):
        distances[i, j] = np.linalg.norm(img[i] - colours[j])
indices = np.argmin(distances, axis=1)
preprocessed_img = colours[indices]
preprocessed_img = preprocessed_img.reshape((32, 32, 3))

plt.imshow(preprocessed_img)
plt.show()

