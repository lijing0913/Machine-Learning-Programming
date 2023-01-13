## K Means Clustering
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import collections

class KMeansClutering:
    def __init__(self):
        self.m = 0

    def check_converge(self, mus: np.array, pre_mus: np.array):
        for x, y in zip(mus, pre_mus):
            if (x == y).all():
                return True
        return False
    
    def assign_clusters(self, X: np.array, mus: np.array):
        clusters = collections.defaultdict(list) # key is cluster index, value is data point
        for x in X:
            dists = [np.linalg.norm(x - mu) for mu in mus]
            min_idx = dists.index(min(dists))
            clusters[min_idx].append(x)
        return clusters
    
    def update_centers(self, mus: np.array, clusters: dict):
        mus = []
        for _, points in clusters.items():
            mus.append(np.mean(points, axis=0))
        return mus

    def fit(self, X:np.array, k:int):
        # Initialize to k random centers
        mus = X[np.random.choice(X.shape[0], k, False)] # generate a uniform random sample from X.shape[0] of size k
        pre_mus = mus + 2
        iter, max_iter = 1, 10

        while iter < max_iter and not self.check_converge(mus, pre_mus):
            pre_mus = mus
            # Assign all points in X to clusters
            clusters = self.assign_clusters(X, mus)
            # Reevaluate centers
            mus = self.update_centers(mus, clusters)
            iter += 1
        return mus, clusters
   
    def predict(self, mus: np.array, X: np.array):
        # Find mean with smallest dist from X
        # if two nums are equidistance from X, return the firsrt one it checked
        min_dist = np.Inf
        best_mu = np.Inf
        for mu in mus:
            dist = np.linalg.norm(X - mu)
            if dist < min_dist:
                min_dist = dist
                best_mu = mu
        return best_mu

if __name__ == "__main__":
    # Generate data
    x1 = np.random.randn(5, 2) + 5
    x2 = np.random.randn(5, 2) - 5
    X = np.concatenate([x1, x2], axis=0) # by column
    k = 2
    kmeans = KMeansClutering()
    mus, clusters = kmeans.fit(X, k)
    print(mus, '\n')
    print(clusters)
    print(kmeans.predict(mus, np.array([6.02, 6.656])))