# coding=utf-8
"""Affinity Propagation clustering algorithm PyTorch implementation."""

import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clustering import Clustering


class AffinityPropagation(Clustering):
    """Affinity Propagation clustering algorithm."""
    def __init__(self, iterations, dumping_factor=0.75):
        self.iterations = iterations
        self.dumping_factor = dumping_factor

    def fit(self, x_data):
        """Find exemplars to cluster data.

        Args:
            x_data: torch tensor.
        """


    def transform(self, x_data):
        """Assign clustering to data.

        Args:
            x_data: torch tensor.

        Returns:
            transformed data (torch tensor).
        """
        # Assign clustering


if __name__ == '__main__':
    print("Generating data...")
    data = torch.from_numpy(make_blobs(50_000)[0])

    print("Clustering data...")
    clustering = AffinityPropagation(500)
    clustering.fit(data)
    assigned_clusters = clustering.transform(data)

    print("Visualizing results...")
    matplotlib.use('webAgg')
    plt.scatter(data[:, 0], data[:, 1], c=assigned_clusters)
    plt.show()
