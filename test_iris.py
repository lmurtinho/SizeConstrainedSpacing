from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from get_scores import get_scores
from pr_clustering import PRClustering
import numpy as np
import argparse

d = load_iris()
data = d['data']
labels = d['target']
k = len(np.unique(labels))

parser = argparse.ArgumentParser(description='PR clustering')
parser.add_argument('--n_init', '-ni', default='square',
                    help='strategy for choosing initial centers')
parser.add_argument('--seed', '-s', type=int, default=None,
                    help='random seed')

if __name__ == '__main__':
    args = parser.parse_args()
    n_init = args.n_init
    seed = args.seed
    km = KMeans(n_clusters=k, random_state=seed)
    km.fit(data)
    km_labels = km.labels_
    print(seed, type(seed))
    if n_init.isdigit():
        if '.' in n_init:
            n_init = float(n_init)
        else:
            n_init = int(n_init)
    print(n_init, type(n_init))
    pr = PRClustering(n_clusters=k, random_state=seed, n_init=n_init)
    pr.fit(data)
    pr_labels = pr.predict(data)

    print('KMeans scores:')
    print(get_scores(data, km_labels, labels))
    print('PRClustering scores:')
    print(get_scores(data, pr_labels, labels))