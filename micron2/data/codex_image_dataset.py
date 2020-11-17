from anndata import AnnData
import numpy as np
import pandas as pd
import h5py


def load_as_anndata(h5data, with_images=True, obsm=None):
  """
  Load a codex image dataset into an AnnData object

  Store the mean intensity values as vars

  keep a hook to the original dataset in order to load raw images

  Args:
    h5data (str, path): file path to hdf5 dataset created by `pull_nuclei()`
    with_images (bool): whether to stash a hook to the original h5data.
                        with_images=True also keeps h5data in an open state
                        (default=True)
    obsm (str): group holding uns datasets

  Returns:
    adata (AnnData)
  """

  h5f = h5py.File(h5data, "r")

  # Make sure all the preprocessing was done
  assert 'meta' in h5f.keys()
  assert 'intensity' in h5f.keys()

  if obsm is not None: 
    assert obsm in h5f.keys()

  # Pull cell IDs, mean intensity features, and channel names
  cell_ids = [b.decode('UTF-8') for b in h5f['meta/Cell_IDs']]
  channel_names = [b.decode('UTF-8') for b in h5f['meta/channel_names']]
  coordinates = h5f['meta/coordinates'][:]
  coordinates[:,1] = -coordinates[:,1] # Flip Y for matplotlib axis conventions
  
  features = np.zeros((len(cell_ids), len(channel_names)))
  for i, channel in enumerate(channel_names):
    vals = h5f[f'intensity/{channel}'][:]
    features[:, i] = vals

  adata = AnnData(features, 
                  obs=pd.DataFrame(index=cell_ids),
                  var=pd.DataFrame(index=channel_names),
                  obsm=dict(coordinates=coordinates),
                  uns=dict(source_data=h5data))

  if obsm is not None:
    obsm_keys = h5f[obsm].keys()
    for k in obsm_keys:
      x = h5f[f'{obsm}/{k}'][:]
      adata.obsm[k] = x

  # adata.uns['images'] is a reference to the "cells" Group in h5data
  h5f.close()
  return adata

