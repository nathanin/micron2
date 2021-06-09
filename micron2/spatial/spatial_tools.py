from micron2.data.pull_nuclei import get_channel_means
from micron2.clustering import cluster_leiden, cluster_leiden_cu
import numpy as np
import pandas as pd
import warnings

from pandas.core.dtypes.common import is_sparse
from scipy.sparse import issparse

from sklearn.neighbors import NearestNeighbors as sk_NearestNeighbors
try:
  import cudf
  from cuml.neighbors import NearestNeighbors
  from cuml.metrics.pairwise_distances import pairwise_distances
  USE_RAPIDS=True
except:
  warnings.warn("Failed to load RAPIDS cuda-enabled tools. Falling back to sklearn")
  USE_RAPIDS=False


__all__ = [
  'get_neighbors',
  'pull_neighbors',
  'categorical_neighbors',
  'sliding_window_niches',
  'sliding_window_values',
  'k_neighbor_niches',
  'celltype_distances'
]

def get_neighbors_rapids(features, k):
  X_cudf = cudf.DataFrame(features)
  # the reference points are included, so add 1 to k
  model = NearestNeighbors(n_neighbors=k+1, output_type='cudf')
  model.fit(features)

  # kn_graph = model.kneighbors_graph(X_cudf)
  distances, indices = model.kneighbors(X_cudf)
  distances = distances.iloc[:, 1:] # Drop the self entry
  indices = indices.iloc[:, 1:]

  distances = distances.to_pandas()
  indices = indices.to_pandas()
  return distances, indices


def get_neighbors_sklearn(features, k):
  # sklearn doesn't include the reference point in the output
  model = sk_NearestNeighbors(n_neighbors=k)
  model.fit(features)
  distances, indices = model.kneighbors(features)
  return pd.DataFrame(distances), pd.DataFrame(indices)
  

def get_neighbors(features, k=5, return_distances=False, backend='rapids' if USE_RAPIDS else 'sklearn'):
  """
  Args:
    features (np.ndarray): (n_cells x n_features)
    k (int): number of neighbors
    return_distances (bool): whether to also return distances
    
  Return:
    indices (cudf.DataFrame)
    distances (cudf.DataFrame)
  """

  if issparse(features):
    features = features.toarray()
  
  if backend=='rapids':
    distance, indices = get_neighbors_rapids(features, k)
  else:
    distance, indices = get_neighbors_sklearn(features, k)

  if return_distances:
    return indices, distance
  else:
    return indices
  


def pull_neighbors(indices , groups , target, mode='mask'):
  """
  Pull indices of all the neighbors belonging to the target group in groups
  
  Note indices refer to 0-indexed cells
  
  Args:
    indices (pandas.DataFrame): (n_cells x n_neighbors) --- should not include self.
    groups (int or categorical): (n_cells) annotations
    target (list, int or str): group ID in groups to focus
    mode (str): ['indices', 'mask'] return indices or a boolean mask
  Returns:
    neighbors
  """
  
  if isinstance(target, list):
    idx = np.sum([groups == t for t in target], axis=0) > 0
  else:
    idx = groups == target
    
  print('getting indices for target', target)
  neighbors = np.unique(indices.loc[idx, :].values.ravel())
  print('found neighbors', neighbors)
  
  if mode == 'indices':
    return neighbors
  elif mode == 'mask':
    idx = np.zeros(len(indices),dtype=bool)
    idx[neighbors] = 1
    return idx



def categorical_neighbors(obs, col1, col2, coords, k=10):
  """ Find the frequency of neighbor cells 
  
  Create a frequency matrix where
  freq ~ (query cells) X (neighbor cells)

  """
  neighbors = np.array(get_neighbors(coords, k=k))
  # print(neighbors.shape)

  query_types = np.array(obs[col1])
  neighbor_types = np.array(obs[col2])
  u_query_types = np.unique(query_types)
  u_neighbor_types = np.unique(neighbor_types)
  neighbor_counts = pd.DataFrame(index=u_query_types, columns=u_neighbor_types, 
                                 dtype=np.int64)

  for q in u_query_types:
    # unique the neighbors or allow them to repeat?
    n = neighbors[query_types==q,:].ravel()
    n = neighbor_types[n]
    n_vals, n_counts = np.unique(n, return_counts=True)
    for t in u_neighbor_types:
      if t in n_vals:
        neighbor_counts.loc[q,t] = n_counts[n_vals==t]
      else:
        neighbor_counts.loc[q,t] = 0

  neighbor_freqs = neighbor_counts.apply(lambda x: x/np.sum(x), axis=1)
  return neighbor_freqs 



def celltype_distances(coords, celltypes, query_cell, target_cell, k=10, mode='nearest', summary_fn=np.mean):
  """
  Get pairwise distances between instances of `query_cell` and `target_cell`

  Modes & interpretations:
    1.  k-nearest: for each instance of `query_cell` find the `k` nearest `target_cells` 
         and return the average distance
    2.  nearest: special case of k-nearest when `k=1`
    3.  overall: summarized by `summary_fn` distance from each instance of `query_cell` to all instances of `target_cell`
    4.  max: distance from each instance of `query_cell` to the furthest instance of `target_cell`

  Default mode: nearest

  Returns the specified distance mode to each instance of `query_cell`. 
  So if there are 100 instances of query cell in `celltypes`, return a 100-length vector
  """
  x = coords[celltypes==query_cell]
  y = coords[celltypes==target_cell]
  D = pairwise_distances(X=x, Y=y, metric='euclidean', output_type='numpy')
  
  if mode == 'overall':
    return summary_fn(D, axis=1)

  if mode in ['k-nearest', 'nearest', 'max']:
    D.sort(axis=1)

    if mode == 'max':
      D = D[:,::-1]
      return D[:,0]
    elif mode == 'nearest':
      return D[:,0]
    elif mode == 'k-nearest':
      return summary_fn(D[:,:k], axis=1)




def sliding_window_niches(coords, clusters, window=100, overlap=0.25, cell_ids=None,
                          aggregate='sum', min_cells=10, verbose=False):
  """

  Returns:
    # phenotypes_cluster (np.int ~ H, W): Niches in a spatial layout
    phenotype_profiles (float ~ H, W, N_phenotypes): Raw phenotype distributions in each window
    window_id (str ~ H, W) ~ Window ID's in a spatial layout
    cell_window_map (dict) ~ Collections of cell IDs within each window
    cluster_levels ~ array mapping positions in `phenotype_profiles` (by index) back to values from `clusters` (by value)
  """
  # swap positions - coords[:,0] ~ width
  # but we want c1 to be height
  c1 = coords[:,1]
  c2 = coords[:,0]

  max_c1, max_c2 = np.max(c1), np.max(c2)
  if verbose:
    print(coords.shape, len(clusters))

  cluster_levels, clusters = np.unique(clusters, return_inverse=True)
  u_clusters = np.unique(clusters)
  w2 = int(window // 2)
  step_size = int(np.floor(window * (1-overlap)))

  if cell_ids is None:
    cell_ids = np.arange(len(clusters))

  centers_c1 = np.arange(w2, max_c1-w2, step_size, dtype=int)
  centers_c2 = np.arange(w2, max_c2-w2, step_size, dtype=int)

  cell_window_map = {}

  n_cells = np.zeros((len(centers_c1), len(centers_c2)))
  phenotype_profiles = np.zeros((len(centers_c1), len(centers_c2), len(u_clusters)))
  # phenotypes_pcts = np.zeros((len(centers_c1), len(centers_c2), len(u_clusters)))
  window_id = np.zeros((len(centers_c1), len(centers_c2)), dtype=object)
  window_id[:] = ''

  if verbose:
    print(f'Checking {len(centers_c1)*len(centers_c2)} windows')
  for i, c1_c in enumerate(centers_c1):
    w_dim1 = [c1_c-w2, c1_c+w2]
    w_cells_dim1 = (c1 > w_dim1[0]) & (c1 < w_dim1[1])
    for j, c2_c in enumerate(centers_c2):
      w_dim2 = [c2_c-w2, c2_c+w2]
      w_cells_dim2 = (c2 > w_dim2[0]) & (c2 < w_dim2[1])
      w_cells = w_cells_dim1 & w_cells_dim2
      
      w_n_cells = np.sum(w_cells)
      n_cells[i,j] = w_n_cells

      if w_n_cells < min_cells:
        continue
      else:
        w_clusters = clusters[w_cells]
        pfl = np.zeros(len(u_clusters), dtype=np.float32)
        # pfl_pct = np.zeros(len(u_clusters), dtype=np.float32)
        for u in u_clusters:
          if aggregate == 'sum':
            pfl[u] = (w_clusters == u).sum()
          elif aggregate == 'percent':
            pfl[u] = (w_clusters == u).sum() / w_n_cells
      
      phenotype_profiles[i,j,:] = pfl
      # phenotypes_pcts[i,j,:] = pfl_pct
      
      w_id = f'window_{i}_{j}'
      window_id[i,j] = w_id
      cell_window_map[w_id] = cell_ids[w_cells]


  return phenotype_profiles, window_id, cell_window_map, cluster_levels




def sliding_window_values(coords, values, window=100, overlap=0.25, cell_ids=None,
                          aggregate='sum', min_cells=10):
  """
  Input:
    coords: (N x 2)
    value: (N x m)

  Returns:
    aggregated_values (float ~ H, W, m): aggregated values in each window
    window_id (str ~ H, W) ~ Window ID's in a spatial layout
    cell_window_map (dict) ~ Collections of cell IDs within each window
  """
  # Coerce values into (N x m) in case a list
  values = np.array(values)
  if len(values.shape) == 1:
    values = np.reshape(values, (-1, 1))

  # swap positions - coords[:,0] ~ width
  # but we want c1 to be height
  c1 = coords[:,1]
  c2 = coords[:,0]

  max_c1, max_c2 = np.max(c1), np.max(c2)

  w2 = int(window // 2)
  step_size = int(np.floor(window * (1-overlap)))

  if cell_ids is None:
    cell_ids = np.arange(values.shape[0])

  centers_c1 = np.arange(w2, max_c1-w2, step_size, dtype=int)
  centers_c2 = np.arange(w2, max_c2-w2, step_size, dtype=int)

  cell_window_map = {}

  n_cells = np.zeros((len(centers_c1), len(centers_c2)))
  window_values = np.zeros((len(centers_c1), len(centers_c2), values.shape[-1]), dtype='float')
  window_id = np.zeros((len(centers_c1), len(centers_c2)), dtype=object)
  window_id[:] = ''

  print(f'Checking {len(centers_c1)*len(centers_c2)} windows')
  for i, c1_c in enumerate(centers_c1):
    w_dim1 = [c1_c-w2, c1_c+w2]
    w_cells_dim1 = (c1 > w_dim1[0]) & (c1 < w_dim1[1])
    for j, c2_c in enumerate(centers_c2):
      w_dim2 = [c2_c-w2, c2_c+w2]
      w_cells_dim2 = (c2 > w_dim2[0]) & (c2 < w_dim2[1])
      w_cells = w_cells_dim1 & w_cells_dim2
      
      w_n_cells = np.sum(w_cells)
      n_cells[i,j] = w_n_cells

      if w_n_cells < min_cells:
        continue
      else:
        w_values = values[w_cells, :]
        if aggregate == 'sum':
          v = w_values.sum(axis=0)
        elif aggregate == 'mean':
          v = w_values.mean(axis=0)
      
      window_values[i,j,:] = v
      # phenotypes_pcts[i,j,:] = pfl_pct
      
      w_id = f'window_{i}_{j}'
      window_id[i,j] = w_id
      cell_window_map[w_id] = cell_ids[w_cells]

  return window_values, window_id, cell_window_map


def k_neighbor_niches(coords, clusters, k=10, max_dist=100, u_clusters=None, aggregate='sum',
                      backend='rapids'):

  if u_clusters is None:
    u_clusters = np.unique(clusters)

  neighbors, distances = get_neighbors(coords, k=k, return_distances=True, backend=backend)
  neighbors = np.array(neighbors.values)
  distances = np.array(distances.values)
  n_cells = neighbors.shape[0]

  profiles = np.zeros((n_cells, len(u_clusters)), dtype=np.float32)

  for i in range(n_cells):
    nbr = neighbors[i]
    if max_dist is not None:
      nbr = nbr[distances[i]<max_dist]

    v = clusters[list(nbr) + [i]] # include self
    p = np.zeros(len(u_clusters))
    for ui, u in enumerate(u_clusters):
      if aggregate=='sum':
        p[ui] = np.sum(v==u)
      elif aggregate=='percent':
        p[ui] = np.sum(v==u)/len(v)

    profiles[i,:] = p

  return profiles