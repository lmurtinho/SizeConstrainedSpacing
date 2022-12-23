import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

class PRClustering():

  def __init__(self, n_clusters, alpha=0.25, 
               n_init = 'square',
               dist_func=lambda x, y: np.linalg.norm(x - y), 
               random_state=None):
    self.n_clusters = n_clusters
    self.d_cluster = bool(n_clusters % 2)
    self.alpha = alpha
    self.dist_func = dist_func
    self.rng = np.random.default_rng(seed=random_state)
    self.n_init = n_init
  
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
    max_dist = 0
    ans = -1
    for i in to_test:
        dist = euclidean_distances(X[i].reshape(1, -1), X).sum()
        # dist = sum([self.dist_func(X[i], X[j]) for j in range(n)])
        if dist > max_dist: 
            ans = i
            max_dist = dist
    return ans
  
  def find_distant_neighbor(self, X, u=None, i=None):
    n = len(X)
    max_dist = -1
    ans = -1
    for v in range(n):
        if v not in self.u_centers + self.v_centers:
          # dist = sum([self.dist_func(X[v], X[j])
          #             for j in self.u_centers + self.v_centers])
          if self.u_centers + self.v_centers:
            dist = euclidean_distances(X[v].reshape(1, -1),
                                        X[self.u_centers + self.v_centers]).sum()
          else:
            dist = 0
          if u is not None:
              factor = self.n_clusters - 2*i + 2
              dist += self.dist_func(X[v], X[u]) * factor
          if dist > max_dist:
              max_dist = dist
              ans = v
    return ans
  
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
    for i in tqdm(range(len(X))):
      if i in self.u_centers:
        labels.append(self.u_centers.index(i))
      elif i in self.v_centers:
        labels.append(self.v_centers.index(i) + len(self.u_centers))
      else:
        labels.append(self.find_label(X, X[i]))
    return np.array(labels, dtype=int)

  def find_label(self, X, v):
    min_dist = np.inf
    label = None
    n_u = len(self.u_centers)
    if not self.d_cluster:
      n_u -= 1
    for i in range(n_u):
      dists_p = [self.dist_func(v, X[self.u_centers[i]]), 
                 self.dist_func(v, X[self.v_centers[i]])]
      min_dist_p = min(dists_p)
      dist_i = self.dist_func(X[self.u_centers[i]], 
                              X[self.v_centers[i]]) * self.alpha
      if (min_dist_p < dist_i) and (min_dist_p < min_dist):
        label = i 
        if dists_p[1] < dists_p[0]:
          label += n_u
    if label is None:
      if self.d_cluster:
        label = 2 * n_u
      else:
        label = i+1
        dists_p = [self.dist_func(v, X[self.u_centers[i+1]]),
                   self.dist_func(v, X[self.v_centers[i+1]])]
        if dists_p[0] > dists_p[1]:
          label += n_u
    return label