import h5py
import tqdm.auto as tqdm
import numpy as np
import warnings

from sklearn.neighbors import NearestNeighbors
try:
  import tensorflow_io as tfio
except:
  warnings.warn('Failed to load tensorflow-io.')

"""
We want to stream sets from hdf5, using tensorflow_io

it's unclear if a py_func based approach that loads
sets live by index could be fast enough to keep the GPU
saturated with data.

To start, create a dataset that's approximately multiplying
the original dataset like:

hdf5/
  cells/
    channel/ <-- (n_cells, n_neighbors, h, w)

"""

def create_set_hdf5(h5f, coords=None, n_neighbors=5, sample_rate=1., outf=None):
  """
  Create a `nuclei_sets` dataset
  
  Args:
    h5f (h5py.File): An open HDF5 file
    coords (np.ndarray): features to use for computing neighbors
    n_neighbors (int): number of neighbors
    sample_rate (float): (0 - 1) percent of data to sample
    outf (str): Path to write hdf5

  Returns:
    None
  """

  NBR = NearestNeighbors(n_neighbors=n_neighbors, 
                         metric='minkowski', p=2)
  NBR.fit(coords)
  nbr = NBR.kneighbors(return_distance=False) # Does not include the query (index) 

  n_cells = nbr.shape[0]
  n_sample = int(n_cells * sample_rate)
  indices = np.random.choice(n_cells, n_sample, replace=False)
  print(n_sample, indices.shape)

  out_h5 = h5py.File(outf, "w")
  datasets = {}
  for c in channel_names:
    d = out_h5.create_dataset(f'cells/{c}', 
                              shape=(n_sample,n_neighbors+1,size,size), 
                              maxshape=(None,n_neighbors+1,size,size),
                              dtype='uint8', 
                              chunks=(1,1,size,size), # ?
                              compression='gzip')
    datasets[c] = d

  pbar = tqdm.tqdm(channel_names)
  for c in pbar:
    pbar.set_description(f'Channel: {c}')
    d = datasets[c]
    for nx, i in enumerate(indices):
      s = stack_neighbors(i, nbr[i], h5f[f'cells/{c}'])
      d[nx,...] = s
    out_h5.flush()

  out_h5.close()

  return None


from functools import lru_cache


def stream_sets(h5f, coords=None, n_neighbors=5, 
                apply=lambda x: x, # Function to apply before yielding
                use_channels=['DAPI', 'CD45', 'PanCytoK'],
                cache_size=30000,
                shuffle=False, seed=999):
  """
  Build sets of individual cells directly from the cell dataset
  
  This function trades speed for flexibility in `n_neighbors`, and on-the-fly neighbor definitions,
  compared to saving a static set of neighborhoods in a `nuclei_sets` dataset type.

  In some trials, increasing cache_size helps with speed, after a large fraction
  of the dataset has been pulled at least once.

  Args:
    h5f: Open HDF5 file
    coords: features to use for nearest neighbors 
    n_neighbors: N neighbors
    use_channels: list of datasets to use (must exist in h5f['cells/<channel>'])
    shuffle (bool): whether to shuffle the images

  Returns:
    iterator
  """

  NBR = NearestNeighbors(n_neighbors=n_neighbors, 
                         metric='minkowski', p=2)
  NBR.fit(coords)
  nbr = NBR.kneighbors(return_distance=False) # Does not include the query (index) 

  n_cells = nbr.shape[0]

  indices = np.arange(n_cells)
  if shuffle:
    np.random.seed(seed)
    np.random.shuffle(indices)


  @lru_cache(maxsize=cache_size)
  def load_image(dataset, index):
    return h5f[dataset][index, ...]

  for i in indices:
    s = []
    for c in use_channels:
      # query_cell = h5f[f'cells/{c}'][i,...] 
      # nbr_cells = [h5f[f'cells/{c}'][j,...] for j in nbr[i]]
      query_cell = load_image(f'cells/{c}', i)
      nbr_cells = [load_image(f'cells/{c}', j) for j in nbr[i]]
      s.append(np.stack([query_cell] + nbr_cells, axis=0))
    s = np.stack(s, axis=-1)
    s = apply(s)
    yield s
