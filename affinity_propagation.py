# coding=utf-8
"""Affinity Propagation clustering algorithm PyTorch implementation."""

from itertools import cycle
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from clustering import Clustering

matplotlib.use('webAgg')


class AffinityPropagation(Clustering):
    """Affinity Propagation clustering algorithm."""
    def __init__(self, iterations, dumping_factor=0.75):
        self.iterations = iterations
        self.dumping_factor = dumping_factor
        self.A = None
        self.R = None
        self.S = None
        self.clustering = None

    @staticmethod
    def similarity(xi, xj):
        """Euclidean distance between vectors.

        Args:
            xi, xj: vectors.

        Returns:
            Euclidean distance.
        """
        return -((xi - xj) ** 2).sum()

    def update_r(self, x_data):
        """Update responsibility matrix."""
        v = self.S + self.A
        rows = np.arange(x_data.shape[0])
        v.fill_diagonal_(-np.inf)
        idx_max = np.argmax(v, axis=1)
        first_max = v[rows, idx_max]
        v[rows, idx_max] = -np.inf

        second_max = v[rows, np.argmax(v, axis=1)]
        max_matrix = torch.zeros_like(self.R) + first_max[:, None]
        max_matrix[rows, idx_max] = second_max
        new_val = self.S - max_matrix

        self.R = self.R * self.dumping_factor + (1 - self.dumping_factor) * new_val

    def update_a(self, x_data):
        """Update availability matrix."""
        k_k_idx = np.arange(x_data.shape[0])
        a = self.R.clone().detach()
        a[a < 0] = 0
        a.fill_diagonal_(0)
        a = a.sum(axis=0)  # columnwise sum
        a = a + self.R[k_k_idx, k_k_idx]

        a = torch.ones(self.A.shape) * a
        a -= np.clip(self.R, 0, np.inf)
        a[a > 0] = 0

        w = self.R.clone().detach()
        w.fill_diagonal_(0)

        w[w < 0] = 0
        a[k_k_idx, k_k_idx] = w.sum(axis=0)  # columnwise sum
        self.A = self.A * self.dumping_factor + (1 - self.dumping_factor) * a

    def plot_iteration(self, x_data):
        """Plot iteration exemplars."""
        fig = plt.figure(figsize=(12, 6))
        sol = self.A + self.R

        labels = np.argmax(sol, axis=1)
        exemplars = np.unique(labels)
        colors = dict(zip(exemplars, cycle('bgrcmyk')))

        for i in range(len(labels)):
            X = x_data[i][0]
            Y = x_data[i][1]

            if i in exemplars:
                exemplar = i
                edge = 'k'
                ms = 10
            else:
                exemplar = labels[i].item()
                ms = 3
                edge = None
                plt.plot([X, x_data[exemplar][0]], [Y, x_data[exemplar][1]], c=colors[exemplar])
            plt.plot(X, Y, 'o', markersize=ms, markeredgecolor=edge, c=colors[exemplar])

        plt.title('Number of exemplars: %s' % len(exemplars))
        return fig, labels, exemplars

    def fit(self, x_data):
        """Find exemplars to cluster data.

        Args:
            x_data: torch tensor.
        """
        self.S = torch.zeros((x_data.shape[0], x_data.shape[0]))
        self.R = self.S.clone().detach()
        self.A = self.S.clone().detach()

        # compute similarity for every data point.
        for i in range(x_data.shape[0]):
            for k in range(x_data.shape[0]):
                self.S[i, k] = self.similarity(x_data[i], x_data[k])

        preference = np.median(self.S)
        self.S.fill_diagonal_(preference)
        damping = 0.5

        figures = []
        last_sol = np.ones(self.A.shape)
        last_exemplars = np.array([])

        c = 0
        for i in range(self.iterations):
            self.update_r(x_data)
            self.update_a(x_data)

            sol = self.A + self.R
            exemplars = np.unique(np.argmax(sol, axis=1))

            if last_exemplars.size != exemplars.size or np.all(last_exemplars != exemplars):
                fig, labels, exemplars = self.plot_iteration(x_data)
                figures.append(fig)

            if np.allclose(last_sol, sol):
                print(exemplars, i)
                break

            last_sol = sol
            last_exemplars = exemplars
            self.clustering = last_exemplars
        plt.show()

    def transform(self, x_data):
        """Assign clustering to data.

        Args:
            x_data: torch tensor.

        Returns:
            transformed data (torch tensor).
        """
        return self.clustering


if __name__ == '__main__':
    print("Generating data...")
    data = torch.from_numpy(make_blobs(500)[0])

    print("Clustering data...")
    clustering = AffinityPropagation(75)
    clustering.fit(data)
