# coding=utf-8
"""K-means clustering algorithm PyTorch implementation."""

import random
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clustering import Clustering


class KMeans(Clustering):
    """K-Means clustering algorithm (Lloyd's algorithm)."""
    def __init__(self, n_clusters, converge_threshold=0.00001):
        self.n_clusters = n_clusters
        self.centroids = None
        self.converge_threshold = converge_threshold

    def fit(self, x_data):
        """Fit centroids to data.

        Args:
            x_data: torch tensor.
        """
        self.centroids = x_data[random.sample(range(len(x_data)), self.n_clusters), :]
        last_change = 1
        iteration = 1
        while last_change > self.converge_threshold:
            print(f"Iteration no. {iteration}")
            iteration += 1
            last_change = 0
            distances = []

            # Assign clustering
            for centroid in self.centroids:
                distances.append(torch.sum(torch.pow((x_data - centroid), 2), dim=1))
            dist = torch.cat(distances).reshape(-1, len(x_data))
            clusters = torch.min(dist, dim=0).indices

            # Move centroids
            new_centers = []
            for center in range(self.n_clusters):
                new_centroid = torch.mean(x_data[clusters == center], dim=0)
                new_centers.append(new_centroid)
                last_change += abs(torch.sum(self.centroids[center, :] - new_centroid))
            self.centroids = torch.cat(new_centers).reshape(self.n_clusters, -1)

    def transform(self, x_data):
        """Assign clustering to data.

        Args:
            x_data: torch tensor.

        Returns:
            transformed data (torch tensor).
        """
        distances = []

        # Assign clustering
        for centroid in self.centroids:
            distances.append(torch.sum(torch.pow((x_data - centroid), 2), dim=1))
        dist = torch.cat(distances).reshape(-1, len(x_data))
        return torch.min(dist, dim=0).indices


if __name__ == '__main__':
    print("Generating data...")
    data = torch.from_numpy(make_blobs(50_000)[0])

    print("Clustering data...")
    clustering = KMeans(5)
    clustering.fit(data)
    assigned_clusters = clustering.transform(data)

    print("Visualizing results...")
    matplotlib.use('webAgg')
    plt.scatter(data[:, 0], data[:, 1], c=assigned_clusters)
    plt.show()
