import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

class PRClustering():

  def __init__(self, n_clusters, alpha=0.25, 
               n_init = 'square',
               dist_func=lambda x, y: np.linalg.norm(x - y),
               min_dist = True,
               random_state=None):
    self.n_clusters = n_clusters
    self.d_cluster = bool(n_clusters % 2)
    self.alpha = alpha
    self.dist_func = dist_func
    self.rng = np.random.default_rng(seed=random_state)
    self.n_init = n_init
    self.min_dist = min_dist
  
  def find_first_point(self, X):
    n = len(X)
    if self.n_init == 'square':
        test_n = int(np.floor(np.sqrt(n)))
    elif self.n_init == 'all':
        test_n = n
    elif type(self.n_init) == int:
        test_n = self.n_init
    elif type(self.n_init) == float:
        test_n = int(np.floor(self.n_init * n))
    else:
        test_n = 1
    to_test = self.rng.choice(n, test_n, replace=False)
    dists = euclidean_distances(X[to_test], X)
    dists_sum = dists.sum(axis=1)
    best_idx = np.argmax(dists_sum)
    self.dists_to_centers = dists[best_idx].reshape(1,-1)
    return to_test[best_idx]
  
  def find_distant_neighbor(self, X, u=None, i=None):
    if not self.u_centers:
      best = self.dists_to_centers.argmax()
    else:
      dists_to_center = self.dists_to_centers.sum(axis=0)
      if u is not None:
        factor = self.n_clusters - 2*i + 2
        dists_to_center += (euclidean_distances(X[u].reshape(1, -1), X) * factor).flatten()
      best = dists_to_center.argmax()
    dists = euclidean_distances(X[best].reshape(1, -1), X)
    self.dists_to_centers = np.concatenate((self.dists_to_centers, dists))
    return best
  
  def fit(self, X, y=None):
    # two "centers" will be selected at each iteration
    
    self.u_centers = []
    self.v_centers = []

    k_ = self.n_clusters // 2
    for i in tqdm(range(k_)):
        if len(self.u_centers) == 0:
           u = self.find_first_point(X)
        else:
           u = self.find_distant_neighbor(X)
        v = self.find_distant_neighbor(X, u, i)
        self.u_centers.append(u)
        self.v_centers.append(v)
    assert len(self.u_centers) == self.n_clusters // 2
    assert len(self.v_centers) == self.n_clusters // 2

  def predict(self, X):
    labels = []
    dists_uv = np.array([self.dist_func(X[self.u_centers[i]],
                                        X[self.v_centers[i]])
                         for i in range(len(self.u_centers))])
    dists_uv *= self.alpha
    for i in tqdm(range(len(X))):
      if i in self.u_centers:
        labels.append(self.u_centers.index(i))
      elif i in self.v_centers:
        labels.append(self.v_centers.index(i) + len(self.u_centers))
      else:
        labels.append(self.find_label(X, X[i], dists_uv))
    return np.array(labels, dtype=int)

  def find_label(self, X, v, dists_uv):
    min_dist = np.inf
    label = None
    n_u = len(self.u_centers)
    if not self.d_cluster:
      n_u -= 1
    for i in range(n_u):
      dists_p = [self.dist_func(v, X[self.u_centers[i]]), 
                 self.dist_func(v, X[self.v_centers[i]])]
      min_dist_p = min(dists_p)
      if (min_dist_p < dists_uv[i]) and (min_dist_p < min_dist):
        if self.min_dist:
          min_dist = min_dist_p
        label = 2 * i
        if dists_p[1] < dists_p[0]:
          label += 1
    if label is None:
      label = 2 * n_u
      if not self.d_cluster:
        dists_p = [self.dist_func(v, X[self.u_centers[i+1]]),
                   self.dist_func(v, X[self.v_centers[i+1]])]
        if dists_p[1] < dists_p[0]:
          label += 1
    return label

  def fit_predict(self, X, y=None):
    self.fit(X)
    return self.predict(X)