import numpy as np

class KMeans():
    def __init__(self, n_clusters, max_iterations=80):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations

    def fit_predict(self, x):
        x = np.array(x, dtype=np.float32)
        k = self.n_clusters
        n = np.size(x, 0)
        np.random.seed(5)

        cluster_centers = x[np.random.choice(range(0, n), k, False)]
        clusters = np.empty((k, 0))
        old_clusters = []

        equal = 0
        iter = 0

        while not equal or iter < self.max_iterations:
            old_clusters = clusters

            cluster_distances = np.zeros((n, k))
            for cluster in range(k):
                cluster_distances[:, cluster] = np.sum(np.sqrt((x - cluster_centers[cluster])**2), 1)

            # "clusters" works as the indexes for the data belonging to each cluster
            clusters = np.argmin(cluster_distances, 1)

            cluster_centers = np.array(
                [np.mean(x[clusters == c], 0) for c in range(k)])

            if old_clusters.tolist() == clusters.tolist():
                equal = 1

            iter += 1

        return clusters