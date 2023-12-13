import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
from Kmeans import KMeans
import os 
import weights
import heapq


import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

def k_nn_features(img):
    features = []
    for i in range(img.shape[0]):
        feature_vector = [img[i][0], img[i][1], img[i][2], i%32, int(i/32)]
        features.append(feature_vector)
    features = np.array(features)
    return features

def k_nn_weight_matrix(img, k):
    img = img.reshape((-1, 3))
    print(img.shape)
    features = k_nn_features(img)
    print(features.shape)
    W = np.zeros((img.shape[0], img.shape[0]))
    neighbors = []
    for i, pixel1 in enumerate(features):
        h = []
        for j, pixel2 in enumerate(features):
            distance = np.linalg.norm(pixel1 - pixel2)
            heapq.heappush(h,  (distance, i, j))
        for i in range(k):
            neighbors.append(heapq.heappop(h))
    for distance, i,  j in neighbors:
        W[i, j] = distance
        W[j, i] = distance
    return W

# horse_images = np.load(os.path.join('Data', 'horse_images.npy'))
# horse_image = horse_images[0]



# W = k_nn_weight_matrix(horse_image, 5)
# print(W)
