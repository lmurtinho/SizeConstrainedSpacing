import networkx as nx
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics.pairwise import euclidean_distances

disable_tqdm = False

def get_data(data_path, filename):
    _, k, _ = filename.split('_')
    df = pd.read_csv(f'{data_path}/{filename}', header=None)
    data = np.array([j for j in df[0].\
                apply(lambda x: np.array([float(i) for i in x.split()]))\
                    .values])
    return data, int(k)

def get_all_dists(data, labels, all_dists=None, verbose=False):
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    ans = np.zeros((n_labels, n_labels))
    for i in tqdm(range(n_labels), disable=not verbose):
        for j in range(i, n_labels):
            if i != j:
                if all_dists is not None:
                    ans[i,j] = ans[j,i] = all_dists[labels==i][:,labels==j].min()
                else:
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

def get_kmeans_losses(data, labels):
    centers = pd.DataFrame(data).groupby(labels).mean()
    dists = [np.square(euclidean_distances(data[labels==i],
                                        centers.iloc[i]\
                                          .values\
                                            .reshape(1,-1))
                    )\
                .sum()
          for i in range(len(centers))]
    return np.array(dists)

def get_cluster_std(labels):
    return np.unique(labels, return_counts=True)[1].std()

def get_scores(data, labels, true_labels=None):
    # vector_dists = euclidean_distances(data)
    dists = get_all_dists(data, labels)
    graph_edges = get_graph_edges(dists)
    mst_edges = get_mst_edges(dists)
    mst_weights = [edge[2]['weight'] for edge in mst_edges]
    graph_weights = [edge[2]['weight'] for edge in graph_edges]
    scores =  {'mst_cost': sum(mst_weights),
            'min_dist': min(mst_weights),
            'kmeans_loss': get_kmeans_losses(data, labels).sum()}
    if true_labels is not None:
        scores['nmi'] = nmi(true_labels, labels)
    return scores

def get_mst_weights(data, labels, all_dists=None):
    """
    Returns the weights of the MST
    of the partition of the data given by the labels.
    """
    dists = get_all_dists(data, labels, all_dists)
    mst_edges = get_mst_edges(dists)
    return [edge[2]['weight'] for edge in mst_edges]

def get_mst_cost(data, labels, all_dists=None):
    """
    Returns the MST cost
    of the partition of the data given by the labels.
    """
    mst_weights = get_mst_weights(data, labels, all_dists)
    return sum(mst_weights)

def get_min_dist(data, labels, all_dists=None):
    """
    Returns the minimum intercluster distance
    of the partition of the data given by the labels.
    """
    mst_weights = get_mst_weights(data, labels, all_dists)
    return min(mst_weights)