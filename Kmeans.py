import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt
import os

def KMeans_pp(img, K, max_iter):
    '''
    2d array as input, output reshaped
    '''
    rows, columns= img.shape
    centroids = [img[np.random.choice(np.arange(rows), 1, replace=False)].squeeze()]
    curr_centroid = centroids[0]
    dist = [[] for i in range(img.shape[0])]
    for _ in range(K-1):
        w = []
        S = 0
        for i, pixel in enumerate(img):
            dist[i].append(np.linalg.norm(curr_centroid - pixel))
            d = min(dist[i])
            S += d
            w.append(S)
        #pick next centroid
        v = np.random.randint(0, S)
        a = 0
        b = len(w)
        while(a < b):
            mid = (a+b)//2
            if(v > w[mid]):
                a = mid + 1
            if(v < w[mid]):
                b = mid - 1
        centroids.append(img[a])
    
    print(centroids)
    centroids = np.array(centroids)
    for i in range(max_iter):
        # Assign each pixel to its closest centroid
        distances = euclidean_distances(img, centroids)
        labels = np.argmin(distances, axis=1) # Along row

        # Update centroids based on the mean pixel value of each cluster
        for j in range(K):
            centroids[j] = np.mean(img[labels == j], axis=0)

    segmented_pixels = centroids[labels]
    segmented_img = segmented_pixels.reshape(img.shape)
    return segmented_img



def KMeans(img, K, max_iter):
    '''
    Only works with 2D numpy arrays
    '''
    rows, columns= img.shape
    centroids = img[np.random.choice(np.arange(rows), K, replace=False)]
    for i in range(max_iter):
        # Assign each pixel to its closest centroid
        distances = euclidean_distances(img, centroids)
        labels = np.argmin(distances, axis=1) # Along row

        # Update centroids based on the mean pixel value of each cluster
        for j in range(K):
            centroids[j] = np.mean(img[labels == j], axis=0)

    segmented_pixels = centroids[labels]
    segmented_img = segmented_pixels.reshape(img.shape)
    return segmented_img

#segmented_image = KMeans(horse_image_reshaped, 4, 2000)
# print(segmented_image)
# plt.imshow(segmented_image)
# plt.show()