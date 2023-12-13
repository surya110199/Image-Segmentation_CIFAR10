import os
import numpy as np
from matplotlib import pyplot as plt
from Kmeans import KMeans
import cv2


def get_location_distances(img):

    coords = [[x, y] for x, y in np.ndindex(img.shape[:2])]

    #print(coords)
    # Print the first 10 coordinates
    coords = np.asarray(coords)


    N = coords.shape[0]
    distances = np.zeros((N, N))
    print(N)
    for i in range(N):
        for j in range(i+1, N):
            # Euclidean distance between pixel i and pixel j
            distances[i, j] = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
            # distances are symmetric, so we can just copy the value over
            distances[j, i] = distances[i, j]
    return distances



# r = 5
# mask = distances < r



def get_hsv(X):
    h, s, v = X[0]/255, X[1]/255, X[2]/255
    output = np.hstack((
        v,
        v * s * np.sin(h),
        v * s * np.cos(h),
    ))
    return output

# X = 32 x 32
def get_weight_matrix(X, distances):
    n = distances.shape[0]
    # X = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)
    X = X.reshape((n, 3))
    W = np.zeros_like(distances)
    
    for i in range(n):
        for j in range(i+1, n):
            if distances[i, j] < 5:
                # distance = np.linalg.norm(get_hsv(X[i]) - get_hsv(X[j]))
                W[i, j] =  np.exp(-np.linalg.norm(get_hsv(X[i]) - get_hsv(X[j])) ** 2/0.01  -  distances[i][j]/ 4)
                W[j, i] = W[i, j]
    return W

# distances = get_location_distances(img)
# W = get_weight_matrix(img, distances)

# print(W.shape)
# D = np.diag(np.sum(W, axis=1))

# L = D - W

# eig_values, eig_vectors = np.linalg.eigh(np.sqrt(np.linalg.inv(D)) @ L @ np.sqrt(np.linalg.inv(D)))

# indices = np.argsort(eig_values)

# U  = eig_vectors[:, :4]

# # # pixel_values = eig_vectors.reshape((-1, 3))

# # # Apply k-means clustering to the pixel values

# # centroids, labels = KMeans(U, 4, 10000)

# # labels = np.argmax(centroids[labels], axis=1)

# # Reshape the labels array back to the original image shape
# segmented_image = KMeans(U, 4, 10000)

# # Assign each cluster a color
# colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]])

# # # Assign each pixel in the segmented image the corresponding color
# color_segmented_image = colors[segmented_image]

# # # Display the segmented image
# plt.imshow(color_segmented_image.astype(np.uint8))
# plt.axis('off')
# plt.show()
