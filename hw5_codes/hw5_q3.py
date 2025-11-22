import numpy as np


class KMeans:
    def __init__(self, n_clusters, max_iter):
        """Initialize a KMeans estimator.

        Parameters
        ----------
        n_clusters : int
            Number of clusters (k).
        max_iter : int
            Maximum number of Lloyd iterations.
        """
        if n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")
        if max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")

        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.cluster_centers_ = None

    def _init_centers(self, X: np.ndarray) -> np.ndarray:
        """Initialize cluster centers by sampling k distinct points from X."""
        n_samples = X.shape[0]
        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot exceed number of samples ({n_samples})."
            )
        idx = np.random.choice(n_samples, size=self.n_clusters, replace=False)
        return X[idx].astype(float, copy=True)

    @staticmethod
    def _pairwise_sq_dists(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute squared Euclidean distances from each point to each center."""
        x2 = np.sum(X**2, axis=1, keepdims=True)            
        c2 = np.sum(centers**2, axis=1, keepdims=True).T    
        cross = X @ centers.T                                
        return x2 + c2 - 2.0 * cross

    def fit(self, X: np.ndarray):
        """Fit KMeans using Lloyd's algorithm.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training samples.

        Returns
        -------
        np.ndarray of shape (n_clusters, n_features)
            Final cluster centers.
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        centers = self._init_centers(X)
        labels = np.full(n_samples, -1, dtype=int)

        for _ in range(self.max_iter):
            dists = self._pairwise_sq_dists(X, centers)
            new_labels = np.argmin(dists, axis=1)

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            new_centers = np.empty_like(centers)
            for k in range(self.n_clusters):
                mask = (labels == k)
                if np.any(mask):
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    j = np.random.randint(0, n_samples)
                    new_centers[k] = X[j]
            centers = new_centers

        self.cluster_centers_ = centers
        self.labels_ = labels
        return self.cluster_centers_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample in X to the nearest learned center.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to label.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted cluster indices in [0, n_clusters-1].
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans has not been fit yet. Call fit(X) first.")
        X = np.asarray(X, dtype=float)
        dists = self._pairwise_sq_dists(X, self.cluster_centers_)
        return np.argmin(dists, axis=1)
