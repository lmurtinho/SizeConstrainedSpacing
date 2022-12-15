from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from get_scores import get_scores
from pr_clustering import PRClustering
import numpy as np

d = load_iris()
data = d['data']
labels = d['target']
k = len(np.unique(labels))
seed = 10

km = KMeans(n_clusters=k, random_state=seed)
km.fit(data)
km_labels = km.labels_

pr = PRClustering(n_clusters=k, random_state=seed)
pr.fit(data)
pr_labels = pr.predict(data)

print('KMeans scores:')
print(get_scores(data, km_labels, labels))
print('PRClustering scores:')
print(get_scores(data, pr_labels, labels))