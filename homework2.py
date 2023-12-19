import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class MNISTDataset:
    def __init__(self, train=True):
        with gzip.open("mnist.pkl.gz", "rb") as fd:
            (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = pickle.load(fd, encoding="latin")

        if train:
            self.inputs = np.concatenate((train_x, test_x))
            self.labels = np.concatenate((train_y, test_y))
        else:
            self.inputs = valid_x
            self.labels = valid_y

mnist_dataset = MNISTDataset(train=True)
X = mnist_dataset.inputs # Shape de forma (70000, 784)

kmeans = KMeans(n_clusters=10, random_state=1234)
kmeans.fit(X)

centroids = kmeans.cluster_centers_

fig, axes = plt.subplots(1, 10, figsize=(20, 4))

for i in range(10):
    centroid_image = centroids[i].reshape(28, 28)  
    axes[i].imshow(centroid_image, cmap='gray')
    axes[i].set_title(f'Centroid {i}')
    axes[i].axis('off')

plt.savefig('centroids.png')
