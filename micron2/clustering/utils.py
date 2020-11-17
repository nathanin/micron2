import leidenalg
import igraph
import numpy as np

from sklearn.neighbors import kneighbors_graph
from matplotlib import pyplot as plt

try:
  import cudf
  import cugraph
  from cuml.neighbors import NearestNeighbors
  import cupy as cp
except:
  print('Failed to import GPU tools. Fast NN will not be possible.')

__all__ = [
  'cluster_leiden',
  'cluster_leiden_cu',
  'run_tsne',
  'plot_embedding'
]

def cluster_leiden(features, neighbors=10, resolution=0.6, n_jobs=8):
  adj = kneighbors_graph(features, n_neighbors=neighbors, n_jobs=n_jobs)
  sources, targets = adj.nonzero()
  edgelist = zip(sources.tolist(), targets.tolist())
  nn_graph = igraph.Graph(edgelist)

  part = leidenalg.find_partition(nn_graph, leidenalg.RBConfigurationVertexPartition,
                                  resolution_parameter=resolution)

  groups = np.array(part.membership)
  return groups


def cluster_leiden_cu(features, neighbors=10, resolution=0.6  ):
  X_cudf = cudf.DataFrame(features)
  model = NearestNeighbors(n_neighbors=neighbors)
  model.fit(features)

  kn_graph = model.kneighbors_graph(X_cudf)
  offsets = cudf.Series(kn_graph.indptr)
  indices = cudf.Series(kn_graph.indices)

  G = cugraph.Graph()
  G.from_cudf_adjlist(offsets, indices, None)
  parts, mod_score = cugraph.leiden(G, resolution=resolution)
  groups = cp.asnumpy(parts['partition'].values)

  groups = np.array([f'{c:02}' for c in groups])
  # groups = np.array([f'{c:02}' for c in groups], dtype='S')
  return groups


def run_tsne(features, train_size=0.2, n_jobs=8):
  """
  Run t-SNE dimensionality reduction on features

  I think we can do initial training on a subset of the samples
  then apply the model in inference-mode to the rest.

  Args:
    features (np.float32): ndarray (n_cells x n_features)

  Returns:
    emb (np.float32): ndarray (n_cells x 2)
  """
  raise NotImplementedError()


def plot_embedding(emb, values, title=None, categorical=False, size=2, ax=None, figsize=(3,3),
                   hideticks=True):
  """
  Scatter plot some cells

  Args:
      emb (np.float32): (n_cells x 2)
      values (np.float32, np.int): (n_cells)
      categorical (bool): Whether to treat values as categorical (i.e. groups)
      ax (matplotlib.pyplot.Axes): axis to use

  Returns:
      -
  """
  if ax is None:
    plt.figure(figsize=figsize)
    ax = plt.gca()
      
  if not categorical:
    srt = np.argsort(values)
    emb = emb[srt,:]
    values = values[srt]
    sp = ax.scatter(emb[:,0], emb[:,1], s=size, c=values)
    plt.colorbar(sp, ax=ax)
  
  else:
    for v in np.unique(values):
      ix = values == v
      ax.scatter(emb[ix, 0], emb[ix, 1], s=size, label=v)
    plt.legend(bbox_to_anchor=(1,1))
  
  if title is not None:
    ax.set_title(title)

  if hideticks:
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
