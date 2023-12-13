import os
from matplotlib import pyplot as plt
import numpy as np

def initialize_centroids(data, k):
    n = data.shape[0]
    indices = np.random.choice(n, k, replace=False)
    centroids = data[indices]
    return centroids

def assign_labels(data, centroids):
    distances = np.sqrt(np.sum((data[:, np.newaxis] - centroids) ** 2, axis=2))
    labels = np.argmin(distances, axis=1)
    return labels

def update_centroids(data, labels, k):
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        centroids[i] = np.mean(data[labels == i], axis=0)
    return centroids

def kmeans(data, k, max_iterations):
    data = data/255
    centroids = initialize_centroids(data, k)
    for _ in range(max_iterations):
        labels = assign_labels(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        segmented_image = centroids[labels]
    return segmented_image

