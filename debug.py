from matplotlib import pyplot as plt
from Kmeans import KMeans, KMeans_pp
import numpy as np
import os 
# from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
import ex
horse_images = np.load(os.path.join('Data', 'horse_images.npy'))
img = horse_images[1000]
plt.imshow(img)
plt.show()
img = img.reshape(1024, 3)
#features = ex.k_nn_features(img)
segmented_img = KMeans_pp(img, 4, 10000)


# segmented_img = segments_slic = slic(img, n_segments=4, compactness=10, sigma=1,
#                      start_label=1) #KMeans(horse_2d, 4, 10000)

plt.imshow(segmented_img[:, :3].
           reshape(32, 32, 3))
plt.show()