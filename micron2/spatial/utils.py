from micron2.data.pull_nuclei import get_channel_means
import numpy as np
import pandas as pd
import warnings

from pandas.core.dtypes.common import is_sparse
from scipy.sparse import issparse

try:
  import cudf
  from cuml.neighbors import NearestNeighbors
  USE_RAPIDS=True
except:
  warnings.warn("Failed to load RAPIDS cuda-enabled tools. Falling back to sklearn")
  from sklearn.neighbors import NearestNeighbors
  USE_RAPIDS=False

__all__ = [
  'get_neighbors',
  'pull_neighbors',
  'categorical_neighbors'
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
  model = NearestNeighbors(n_neighbors=k)
  model.fit(features)
  distances, indices = model.kneighbors(features)
  return distances, indices
  

def get_neighbors(features, k=5, return_distances=False):
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
  
  if USE_RAPIDS:
    distance, indices = get_neighbors_rapids(features, k)
  else:
    distance, indices = get_neighbors_sklearn(features, k)

  if return_distances:
    return indices, distances
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
  """ Find the frequency of neighbor cells """
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