from anndata import AnnData
import numpy as np
import pandas as pd
import h5py
import warnings

from scipy.sparse import csr_matrix

# For deserializing tile / nuclei relationship dictionary
# https://stackoverflow.com/a/48101771
import ast


def load_as_anndata(h5data, obs_names='meta/Cell_IDs', 
                    featurekey = 'cell_intensity',
                    membrane_featurekey = None,
                    coordkey = 'meta/cell_coordinates',
                    flip_y=True, 
                    reverse_coords=False,
                    subtract_min_coords=True, # place the top-left (bottom-left) cell at (0,0)
                    with_images=True,  # unused
                    recover_tile_nuclei=True,
                    keep_open=False,
                    as_sparse=True,
                    obsm=None):
  """
  Load a codex image dataset into an AnnData object

  Store the mean intensity values as vars

  keep a hook to the original dataset in order to load raw images

  Args:
    h5data (str, path): file path to hdf5 dataset created by `pull_nuclei()`
    coordkey (str): key in HDF5 to use as (x,y) coordinates
    with_images (bool): whether to stash a hook to the original h5data.
                        with_images=True also keeps h5data in an open state
                        (default=True) (TODO)
    keep_open (bool): keep the HDF5 file open in "r+" mode and return 
                      an hook to the open file in adata.uns['HDF5']
                      -----> Is this safe for saving via adata.write() ??? <------
    obsm (str): group holding uns datasets

  Returns:
    adata (AnnData)
  """

  h5f = h5py.File(h5data, "r")

  # Make sure all the preprocessing was done
  assert 'meta' in h5f.keys()
  assert featurekey in h5f.keys()

  # if obsm is not None: 
  #   assert obsm in h5f.keys()

  # Pull coordinates, cell IDs, mean intensity features, and channel names
  coordinates = h5f[coordkey][:]

  if subtract_min_coords:
    coordinates[:,0] = coordinates[:,0] - min(coordinates[:,0])
    coordinates[:,1] = coordinates[:,1] - min(coordinates[:,1])

  # As long as coordinates, exist, even if other fields dont exist, return an AnnData
  if obs_names in h5f.keys():
    cell_ids = [b.decode('UTF-8') for b in h5f[obs_names]]
  else:
    # Call them objects
    cell_ids = [f'object_{i}' for i in range(coordinates.shape[0])]

  channel_names = [b.decode('UTF-8') for b in h5f['meta/channel_names']]


  # ----------------------------- Build OBSM -----------------------------------
  # ## These options are here because of inconsistency in the nuclei-centered and
  # ## tile-grid methods of pulling data.
  # if reverse_coords:
  #   coordinates = coordinates[:,::-1]

  if flip_y:
    coordinates[:,1] = -coordinates[:,1] # Flip Y for matplotlib axis conventions

  obsm_dict = dict(coordinates=coordinates)
  # ----------------------------- / Build OBSM -----------------------------------
  
  features = np.zeros((len(cell_ids), len(channel_names)))

  # Check for sameness of length btw features and "cells"
  vals = h5f[f'{featurekey}/{channel_names[0]}'][:]
  if vals.shape[0] == len(cell_ids):
    for i, channel in enumerate(channel_names):
      vals = h5f[f'{featurekey}/{channel}'][:]
      features[:, i] = vals
  else:
    warnings.warn(f'Values ({vals.shape[0]}) mismatch cell ids ({len(cell_ids)})')


  if membrane_featurekey is not None:
    membrane_features = np.zeros((len(cell_ids), len(channel_names)))
    vals = h5f[f'{membrane_featurekey}/{channel_names[0]}'][:]
    if vals.shape[0] == len(cell_ids):
      for i, channel in enumerate(channel_names):
        vals = h5f[f'{membrane_featurekey}/{channel}'][:]
        membrane_features[:, i] = vals
    else:
      warnings.warn(f'Membrane feature values ({vals.shape[0]}) mismatch cell ids ({len(cell_ids)})')
    features = np.concatenate([features, membrane_features], axis=1)

  # ----------------------------- Build UNS -----------------------------------
  uns_dict = dict(source_data=h5data)
  if recover_tile_nuclei:
    """
    Load the dictionary that describes the relationship between tiles and the
    nuclei found within those tiles

    REF https://stackoverflow.com/a/48101771
    """
    tile_nuclei = ast.literal_eval(h5f['images'].attrs['tile_encapsulated_cells'])
    uns_dict['tile_nuclei'] = tile_nuclei

  # uns_dict['image_sources'] = ast.literal_eval(h5f['meta'].attrs['image_sources'])

  # ----------------------------- / Build UNS -----------------------------------

  feature_names = channel_names
  if membrane_featurekey is not None:
    for feature_name in feature_names:
      feature_names.append(f'{feature_name}_membrane')

  print(f'Features are {100*(features==0).sum()/np.prod(features.shape):2.2f}% zeros')

  adata = AnnData(csr_matrix(features) if as_sparse else features, 
                  obs=pd.DataFrame(index=cell_ids),
                  var=pd.DataFrame(index=channel_names),
                  obsm=obsm_dict,
                  uns=uns_dict)

  # if obsm is not None:
  #   obsm_keys = h5f[obsm].keys()
  #   for k in obsm_keys:
  #     x = h5f[f'{obsm}/{k}'][:]
  if obsm is not None:
    if not isinstance(obsm, list):
      obsm = [obsm]

    for k in obsm:
        adata.obsm[k] = h5f[k][:]

  # adata.uns['images'] is a reference to the "cells" Group in h5data
  h5f.close()
  return adata

