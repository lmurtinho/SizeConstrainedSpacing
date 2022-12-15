import numpy as np

class PRClustering():

  def __init__(self, n_clusters, alpha=0.25, 
               dist_func=lambda x, y: np.linalg.norm(x - y), 
               random_state=None):
    self.n_clusters = n_clusters
    self.alpha = alpha
    self.dist_func = dist_func
    self.rng = np.random.default_rng(seed=random_state)
  
  def find_first_point(self, X):
    n = len(X)
    test_n = int(np.floor(np.sqrt(n)))
    to_test = self.rng.integers(0, n, test_n)
    max_dist = 0
    ans = -1
    for i in to_test:
      dist = sum([self.dist_func(X[i], X[j]) for j in range(n)])
      if dist > max_dist:
        ans = i
        max_dist = dist
    return ans
  
  def find_distant_neighbor(self, X, u=None, i=None):
    n = len(X)
    max_dist = -1
    ans = -1
    for v in range(n):
        dist = sum([self.dist_func(X[v], X[j])
                    for j in self.u_centers + self.v_centers])
        if u is not None:
            factor = self.n_clusters - 2*i + 2
            dist += self.dist_func(X[v], X[u]) * factor
        if dist > max_dist:
            max_dist = dist
            ans = i
    return ans
  
  def fit(self, X, y=None):
    # two "centers" will be selected at each iteration
    k_ = self.n_clusters // 2
    
    self.u_centers = []
    self.v_centers = []

    for i in range(1, k_+1):
        if len(self.u_centers) == 0:
           u = self.find_first_point(X)
        else:
           u = self.find_distant_neighbor(X)
        v = self.find_distant_neighbor(X, u, i)
        self.u_centers.append(u)
        self.v_centers.append(v)

  def predict(self, X):
    return [self.find_label(X, v) for v in X]
  
  def find_label(self, X, v):
    min_dist = np.inf
    label = None
    n_u = len(self.u_centers)
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
      label = 2 * n_u
    return label