import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering

class FeasibleSpacing():

    def __init__(self, n_clusters, min_size, verbose=False):
        """
        Algorithm to find a partition into n_clusters clusters
        by optimizing the minimum spacing, provided that all
        clusters in the partition have at least min_size elements.
        """
        self.n_clusters = n_clusters
        self.min_size = min_size
        self.verbose = verbose

    def fit_predict(self, X, sl_model=None):
        """
        Fit the model and returns the labels of each element in X.
        sl_model is a fitted scikit-learn AgglomerativeClustering model
        using single linkage. If sl_model is not provided, the method
        fits a single-linkage clustering model, which may take some time.
        """
        if sl_model is None:
            if self.verbose:
                print("No single linkage model provided, fitting the model.")
            sl_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage='single')
            sl_model.fit(X)
        if self.verbose:
            print('get cluster distribution')
        n = len(X)
        clusters = self.get_cluster_distribution(sl_model.children_, n)
        if self.verbose:
            print('find smallest number of clusters needed')
        min_clusters = self.find_n(clusters, n+1, self.n_clusters, len(clusters))
        if self.verbose:
            print('get labels')
        labels = self.get_labels(X, sl_model, min_clusters)
        if self.verbose:
            print('agglomerate clusters')
        n_per_cluster = np.unique(labels, return_counts=True)[1]
        _, clusters = self.agglomerate_clusters(n_per_cluster, n+1, track_clusters=True)
        self.labels_ = clusters[labels]
        return self.labels_

    def get_cluster_distribution(self, tree, n):
        """
        Returns the number of elements per cluster at each level of the tree.
        At the first level, each element is a singleton, so there are n clusters with 1 element each.
        At the last level, all elements are in a single cluster, so there is 1 cluster with n elements.
        """
        n_per_cluster = [1] * n
        clusters = [[1] * n]
        for i in tqdm(tree, disable=not self.verbose):
            c1, c2 = i
            n_per_cluster.append(n_per_cluster[c1] + n_per_cluster[c2])
            n_per_cluster[c1] = n_per_cluster[c2] = 0
            clusters.append([i for i in n_per_cluster if i > 0])
        return clusters

    def agglomerate_clusters(self, s, max_val, track_clusters=False):
        """
        Agglomerates the clusters in s into k clusters.
        s is an array showing how many elements are in each cluster.
        k is the number of clusters to agglomerate into.
        vals is the answer array, with k clusters.
        At each step, the smallest cluster in s is merged into the smallest cluster in vals.
        If track_clusters is True, also returns an array with the same size as s,
        showing to which cluster in vals each cluster in s was merged.
        """
        vals = np.zeros(self.n_clusters, dtype=int)
        if track_clusters:
            corresp = np.full(len(s), -1, dtype=int)
        while s[s!=max_val].sum() > 0:
            s_idx = s.argmin()
            val = s[s_idx]
            s[s_idx] = max_val
            v_idx = vals.argmin()
            vals[v_idx] += val
            if track_clusters:
                corresp[s_idx] = v_idx
        if track_clusters:
            return vals, corresp
        else:
            return vals

    def find_n(self, clusters, max_val, n_min, n_max):
        """
        Finds the smallest number of clusters that can be agglomerated into k clusters,
        such that each of the k clusters has at least min_size elements.
        """
        if n_max - n_min == 1:
            if self.verbose:
                print(n_min, n_max)
            s = np.array(sorted(clusters[-n_min]))
            check = self.agglomerate_clusters(s, max_val)
            if check.min() >= self.min_size:
                return n_min
            else:
                return n_max
        n = (n_max - n_min) // 2 + n_min
        if self.verbose:
            print(n_min, n_max, n)
        s = np.array(sorted(clusters[-n]))
        check = self.agglomerate_clusters(s, max_val)
        if check.min() >= self.min_size:
            return self.find_n(clusters, max_val, n_min, n)
        else:
            return self.find_n(clusters, max_val, n, n_max)

    def get_labels(self, X, model, n_clusters):
        """
        Returns the labels of the clusters in model, with n_clusters clusters.
        """
        labels = np.arange(len(X), dtype=int)
        for i in tqdm(range(len(model.children_)-n_clusters+1),
                    disable=not self.verbose):
            c1, c2 = model.children_[i]
            labels[labels==c1] = len(X) + i
            labels[labels==c2] = len(X) + i
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            labels[labels == label] = i
        return labels