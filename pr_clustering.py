import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

def euclid_dist():
  pass

class PRClustering():

  def __init__(self, n_clusters, alpha=0.25, 
               n_init = 'square',
               random_state=None,
               euclid_dist=None,
               avoid_small_clusters=True):
    self.n_clusters = n_clusters
    self.n_odd = bool(n_clusters % 2)
    self.alpha = alpha
    self.rng = np.random.default_rng(seed=random_state)
    self.n_init = n_init
    self.euclid_dist = euclid_dist
    self.avoid_small_clusters = avoid_small_clusters
  
  def find_first_point(self, X):
    n = len(X)
    if self.n_init == 'square':
        test_n = int(np.floor(np.sqrt(n)))
    elif self.n_init == 'all':
        test_n = n
    elif type(self.n_init) == int:
        test_n = min(self.n_init, n)
    elif type(self.n_init) == float:
        test_n = min(int(np.floor(self.n_init * n)), n)
    else:
        test_n = 1
    to_test = self.rng.choice(n, test_n, replace=False)
    dists = euclidean_distances(X[to_test])
    idx_a, idx_b = dists.argmax() % test_n, dists.argmax() // test_n
    a, b = to_test[idx_a], to_test[idx_b]
    dists = euclidean_distances(X[[a, b]], X)
    best = dists.argmax() % n
    self.dists_to_centers = euclidean_distances(X[best].reshape(1, -1), X)
    self.dists_to_centers[:,best] = 0
    return best
  
  def find_distant_neighbor(self, X, u=None, i=None):
    if not self.u_centers:
      dists_to_center = self.dists_to_centers
    else:
      dists_to_center = self.dists_to_centers.sum(axis=0)
      if u is not None:
        factor = self.n_clusters - 2*i + 2
        ed = (euclidean_distances(X[u].reshape(1, -1), X) * factor).flatten()
        ed[self.u_centers + self.v_centers + [u]] = 0
        dists_to_center += ed
    best = dists_to_center.argmax()
    dists = euclidean_distances(X[best].reshape(1, -1), X)
    self.dists_to_centers = np.concatenate((self.dists_to_centers, dists))
    self.dists_to_centers[:,best] = 0
    for i in self.u_centers + self.v_centers:
      self.dists_to_centers[:,i] = 0
    if u:
      self.dists_to_centers[:,u] = 0
    return best
  
  def fit(self, X, y=None):
    # two "centers" will be selected at each iteration
    
    self.u_centers = []
    self.v_centers = []

    k_ = (self.n_clusters - 1) // 2
    for i in tqdm(range(k_)):
        if len(self.u_centers) == 0:
           u = self.find_first_point(X)
        else:
           u = self.find_distant_neighbor(X)
        v = self.find_distant_neighbor(X, u, i)
        self.u_centers.append(u)
        self.v_centers.append(v)
    if not self.n_odd:
      u = self.find_distant_neighbor(X)
      self.u_centers.append(u)
    if self.n_odd:
      assert len(self.u_centers) == k_
    else:
      assert len(self.u_centers) == k_ + 1
    assert len(self.v_centers) == k_
    del self.dists_to_centers

  def predict(self, X):
    labels = []
    closest_centroids = []
    centers = self.u_centers + self.v_centers
    dists_uv = euclidean_distances(X[centers])
    dists_uv *= self.alpha
    for i in tqdm(range(len(X))):
      if i in self.u_centers:
        labels.append(self.u_centers.index(i))
      elif i in self.v_centers:
        labels.append(self.v_centers.index(i) + len(self.u_centers))
      else:
        cluster, closest = self.find_label(X, X[i], centers, dists_uv)
        labels.append(cluster)
        closest_centroids.append(closest)
    labels = np.array(labels, dtype=int)
    self.labels_ = labels
    return labels

  def find_label(self, X, x, centers, dists_uv):
    min_dist = np.inf
    label = None
    n_u = len(self.u_centers)
    n_v = len(self.v_centers)

    dists_x = euclidean_distances(X[centers],
                                     x.reshape(1, -1)).flatten()
    idx = dists_x.argmin()

    # find closest cluster
    if idx < n_u:
      closest = 2 * idx
    else:
      closest = 2 * (idx % n_u) + 1

    # check if x should go to closest cluster or to last cluster
    min_dist = dists_x.min()
    dists_x -= min_dist
    one_min = np.isclose(dists_x, 0).sum()
    all_distant = np.all(dists_x >= dists_uv[idx])
    if one_min and all_distant:
      label = closest
    else:
      label = n_u + n_v

    return label, closest

  def fit_predict(self, X, y=None):
    self.fit(X)
    return self.predict(X)