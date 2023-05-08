from sklearn.cluster import AgglomerativeClustering
from feasible import FeasibleSpacing
import numpy as np
from get_scores import get_mst_cost, get_min_dist

class ConstrainedMaxMST():

    def __init__(self, n_clusters, min_size, factor=0.75, verbose=False, random_state=None,
                 store_costs=True):
        """
        Algorithm to find a partition into n_clusters clusters
        by optimizing the minimum spanning tree cost, provided that all
        clusters in the partition have at least min_size * factor elements.
        """
        self.n_clusters = n_clusters
        self.min_size = min_size
        self.factor = factor
        self.verbose = verbose
        self.random_state = random_state
        self.store_costs = store_costs
        if store_costs:
            self.min_dists = []
    
    def fit(self, X, sl_model=None):
        """
        Fits the model. For each possible k from 2 to n_clusters, it fits a
        FeasibleSpacing model with k clusters and stores the labels. Then it
        turns this k-clustering into a partition with n_clusters, and stores
        all of this partitions.
        """
        if sl_model is None:
            if self.verbose:
                print("No single linkage model provided, fitting the model.")
            sl_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=0,
                                               linkage='single')
            sl_model.fit(X)
        ks = range(2, self.n_clusters+1)
        self.labels_for_k = {k: self.fit_for_k(X, k, sl_model)
                             for k in ks}
        if self.store_costs:
            self.min_dist_sum = sum(self.min_dists)
    
    def predict(self, X):
        """
        Returns the partition with the best MST cost among all partitions
        found by the `fit` method.
        """
        ks = range(2, self.n_clusters+1)
        mst_cost_for_k = {k: get_mst_cost(X, self.labels_for_k[k])
                          for k in ks}
        if self.store_costs:
            self.mst_cost_for_k = mst_cost_for_k
        self.best_k = max(mst_cost_for_k, key=mst_cost_for_k.get)
        self.labels_ = self.labels_for_k[self.best_k]
        return self.labels_

    def fit_for_k(self, X, k, sl_model, fs_model=None):
        """
        Finds a FeasibleSpacing k-clustering and adapts it to
        a a ConstrainedMaxMST clustering with n_clusters.
        """
        if self.verbose:
            print(f'Fitting for k={k}')
        if fs_model is None:
            if self.verbose:
                print("No feasible spacing model provided, fitting the model.")
            fs_model = FeasibleSpacing(n_clusters=k,
                                       min_size=self.min_size,
                                       factor=self.factor,
                                       verbose=self.verbose)
            labels = fs_model.fit_predict(X, sl_model)
        else:
            labels = fs_model.labels_
        if self.store_costs:
            self.min_dists.append(get_min_dist(data, labels))

        counts = np.bincount(labels)
        non_visited = np.argsort(counts)[::-1]
        len_non_visited = len(non_visited)
        if self.verbose:
            print('counts:', counts)

        new_labels = np.full(len(labels), -1, dtype=int)
        n_new_labels = 0

        for i in range(len(non_visited)):
            current = non_visited[i]
            size_current = counts[current]
            labels_to_split = np.where(fs_model.labels_ == current)[0]

            rng = np.random.default_rng(seed=self.random_state)
            labels_to_split = rng.permutation(labels_to_split)

            len_non_visited -= 1 # TODO: check if this is correct 
            n_groups = self.find_n_groups(n_new_labels, len_non_visited, size_current)

            splits = self.get_splits(labels_to_split, n_groups)
            for i in range(n_groups):
                new_labels[splits[i]] = n_new_labels
                n_new_labels += 1
        if self.verbose:
            print('new counts:', np.bincount(new_labels))
            print()
        return new_labels
    
    def find_n_groups(self, n_new_clusters, n_old_clusters, size_current):
        """
        Finds the number of groups to split the current cluster into.
        - If dividing the current cluster into n_groups clusters of size
            min_size is not enough to reach n_clusters clusters,
            then it divides the current cluster into n_groups clusters of
            size min_size.
        - If that's not the case, it divides the current cluster into as many
            clusters as needed to leave 1 cluster for each of the remaining
            clusters.
        """
        # TODO: check if min_size must be multiplied by factor here
        min_size = max(int(self.factor * self.min_size), 1)
        if n_new_clusters + n_old_clusters + (size_current // min_size) < self.n_clusters:
            n_groups = size_current // min_size
            if self.verbose:
                print(f'{n_groups} groups of type A')
        else:
            n_groups = self.n_clusters - n_new_clusters - n_old_clusters
            if self.verbose:
                print(f'{n_groups} groups of type B')
        return n_groups
    
    def get_splits(self, labels_to_split, n_groups):
        """
        Splits the labels in labels_to_split into n_groups balanced groups.
        If perfect balance is impossible, distributes the remainder as
        equally as possible.
        """
        remainder = len(labels_to_split) % n_groups
        if self.verbose:
            print(len(labels_to_split), n_groups, remainder)
        if remainder:
            splits = np.split(labels_to_split[:-remainder], n_groups)
            for i in range(remainder):
                splits[i] = np.append(splits[i], labels_to_split[-i-1])
        else:
            splits = np.split(labels_to_split, n_groups)
        assert len(splits) == n_groups
        assert sum([len(s) for s in splits]) == len(labels_to_split)
        return splits