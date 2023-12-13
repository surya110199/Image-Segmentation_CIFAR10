import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
from Kmeans import KMeans, KMeans_pp
import ex
import os 
import weights
import copy
import Kmeans_pp

if __name__ == '__main__':

    deer_images = np.load(os.path.join('Data', 'horse_images.npy'))
    horse_image = deer_images[1000]

    plt.imshow(horse_image.reshape(32, 32, 3))
    plt.show()

    h, w, d = horse_image.shape

    horse_image_reshaped = copy.deepcopy(horse_image.reshape((w*h, d))) 

    K = 4 # ---> No of clusters

    weight_matrix = ex.k_nn_weight_matrix(horse_image, 5)# weights.get_weight_matrix(horse_image, weights.get_location_distances(horse_image)) ## 
    degree_matrix = np.diag(np.matmul(weight_matrix, np.ones((w*h))))

    laplacian = degree_matrix - weight_matrix
    D = degree_matrix
    L = laplacian
    normalized_laplacian = np.sqrt(np.linalg.inv(D)) @ L @ np.sqrt(np.linalg.inv(D))
    #print(laplacian)
    v, w = np.linalg.eigh(normalized_laplacian)
    U_t = np.array(w[:, :K])
    for i in range(len(U_t)):
        U_t[i] = U_t[i]/np.linalg.norm(U_t[i])
    segmented_image = KMeans(U_t, K, 10000)
    print(segmented_image.shape)
    classes = np.unique(segmented_image, axis = 0)
    for i, v in enumerate(segmented_image):
        #print(f"vector {v}")
        if( np.array_equal(v, classes[0]) ):
            horse_image_reshaped[i] = np.array([0, 0, 0])
        if(np.array_equal(v, classes[1])):
            horse_image_reshaped[i] = np.array([252, 0, 0])
        if(np.array_equal(v, classes[2])):
            horse_image_reshaped[i] = np.array([0, 252, 0])
        if(np.array_equal(v, classes[3])):
            horse_image_reshaped[i] = np.array([0, 0, 252])

    plt.imshow(horse_image_reshaped.reshape(32, 32, 3))
    plt.show()
