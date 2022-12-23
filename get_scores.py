from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics.pairwise import euclidean_distances
import networkx as nx
import numpy as np
from tqdm import tqdm

def get_all_dists(data, labels):
  unique_labels = np.unique(labels)
  n_labels = len(unique_labels)
  ans = np.zeros((n_labels, n_labels))
  for i in tqdm(range(n_labels)):
    for j in range(i, n_labels):
      if i != j:
        ans[i,j] = ans[j,i] = euclidean_distances(data[labels==i],
                                                  data[labels==j]).min()
  return ans

def get_mst_edges(dists):
  g = nx.from_numpy_array(dists)
  mst = nx.minimum_spanning_tree(g)
  assert mst.number_of_edges() == g.number_of_nodes() - 1
  assert mst.number_of_nodes() == g.number_of_nodes()
  return mst.edges.data()

def get_graph_edges(dists):
  g = nx.from_numpy_array(dists)
  return g.edges.data()

def get_scores(data, labels, true_labels=None):
  # vector_dists = euclidean_distances(data)
  dists = get_all_dists(data, labels)
  graph_edges = get_graph_edges(dists)
  mst_edges = get_mst_edges(dists)
  mst_weights = [edge[2]['weight'] for edge in mst_edges]
  graph_weights = [edge[2]['weight'] for edge in graph_edges]
  scores =  {'mst_cost': sum(mst_weights),
            'min_dist': min(mst_weights),
            'avg_dist': sum(graph_weights) / len(graph_weights),
            'sil_score': silhouette_score(data, labels),
            'ch_score': calinski_harabasz_score(data, labels),
            'db_score': davies_bouldin_score(data, labels)}
  if true_labels is not None:
    scores['nmi'] = nmi(true_labels, labels)
  return scores