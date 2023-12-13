import tarfile
import numpy as np
import pickle

# Extract the CIFAR-10 tarfile
# tar = tarfile.open('cifar-10-python.tar.gz', 'r:gz')
# tar.extractall()
# tar.close()

# Load the training dataset
train_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
train_data = []
train_labels = []
for batch_name in train_batches:
    with open(f'Data\cifar-10-batches-py\{batch_name}', 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        train_data.append(batch[b'data'])
        train_labels += batch[b'labels']

# Concatenate the training data into a single array
train_data = np.concatenate(train_data, axis=0)

# Reshape the flattened image vectors into their original 32x32x3 shape
train_data = train_data.reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))

# Create boolean masks for horse, deer, and airplane classes
horse_mask = np.array(train_labels) == 7
deer_mask = np.array(train_labels) == 4
airplane_mask = np.array(train_labels) == 0

# Apply the boolean masks to the training data
horse_images = train_data[horse_mask]
deer_images = train_data[deer_mask]
airplane_images = train_data[airplane_mask]


np.save("Data\Horse_images.npy", horse_images)
np.save("Data\Deer_images.npy", deer_images)
np.save("Data\Airplane_images.npy", airplane_images)