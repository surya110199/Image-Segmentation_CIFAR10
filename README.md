# Image segmentation using kmeans and spectral clustering.

This project is mainly about a comparative analysis for performing image segmentation on the CIFAR-10 dataset using K Means and Spectral Clustering techniques.

## Dataset:-
For this project, I only used three classes namely Airplane, Deer, and Horses of the CIFAR-10 dataset. The segmented images can be found in the report_images folder.

## Code Description:-
1) The raw data can be found under the Data Folder. I extracted Airplane, deer, and horse images and stored them in .npy files.
2) K means.py is the code for Image segmentation using K means Clustering
3) weights.py is the code for generating weights for the Adjacency matrix using Shi-Malik and Ng-Spectral clustering.
4) spectral_clustering.py is the code for spectral clustering.

## How the code works:-
For kmeans results:- Run Kmeans.py and debug.py to see the results. You can change the images in debug.py to see other class image segmentation results.
For Spectral clustering:- Run weights.py and Kmeans.py first and then run spectral_clustering.py. Finally, you can visualize the segmentation results using debug.py.

## References:-
1) CIFAR-10 Dataset:- (https://www.cs.toronto.edu/~kriz/cifar.html)
2) Jianbo Shi and J. Malik. Normalized cuts and image segmentation. IEEE Trans. on Pattern Analysis and Machine Intelligence, 22(8):888–905, 2000.
3) Andrew Y. Ng, Michael I. Jordan, and Yair Weiss. On spectral clustering: Analysis and an algorithm. In ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, pages 849–856. MIT Press, 2001.


