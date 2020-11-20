import numpy as np
import pandas as pd

import cudf
import cugraph
from cuml.neighbors import NearestNeighbors
import cupy as cp

__all__ = [
  'get_neighbors',
  'pull_neighbors'
]

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
  
  X_cudf = cudf.DataFrame(features)
  # the reference points are included, so add 1 to k
  model = NearestNeighbors(n_neighbors=k+1)
  model.fit(features)

  # kn_graph = model.kneighbors_graph(X_cudf)
  distances, indices = model.kneighbors(X_cudf)
  distances = distances.iloc[:, 1:] # Drop the self entry
  indices = indices.iloc[:, 1:]
  
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
    
  neighbors = np.unique(indices.loc[idx, :].values.ravel())
  
  if mode == 'indices':
    return neighbors.get()
  elif mode == 'mask':
    idx = np.zeros(len(indices),dtype=bool)
    idx[neighbors.get()] = 1
    return idx