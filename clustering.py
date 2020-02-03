# coding=utf-8
"""Abstract class for clustering algorithms."""

from abc import ABC, abstractmethod


class Clustering(ABC):
    """Abstract clustering class."""
    @abstractmethod
    def fit(self, x_data):
        """Fit clustering to data.

        Args:
            x_data: torch tensor with data to fit.
        """
        pass

    @abstractmethod
    def transform(self, x_data):
        """Transform data according to fitted clustering.

        Args:
            x_data: torch tensor to be transformed.

        Returns:
            transformed data (torch tensor).
        """
        pass
