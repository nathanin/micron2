import scanpy as sc
import pandas as pd
import numpy as np

import glob
import os

import seaborn as sns
from scipy.sparse import issparse
from matplotlib.colors import rgb2hex
from micron2.spatial import get_neighbors, pull_neighbors




def get_channel_image_path(data_dir, channel):
  """ Search for an image corresponding to each channel """
  image_paths = sorted(glob.glob(f'{data_dir}/images/*.tif'))
  # TODO Fix handling DAPI
  for p in image_paths:
    if channel == 'DAPI':
      srch = f'_{channel}'
    else:
      srch = f'_{channel}_'
    if srch in p:
      return p
  # If no image is found for the requested channel, return None
  return None


def set_active_slide(adata_path, shared_variables, logger):
  """ Load a sample's AnnData representation and populate variables

  Assume directory structure:

  full_sample_id/
    images/
      full_sample_id_DAPI.tif
      full_sample_id_CD8.tif
      ...

    full_sample_id.hdf5 <-- input
    full_sample_id_nuclei.tif
    
  Assume an images/ directory exists next to the provided adata_path.

  We need to load the anndata object and populate these variables:

  If an image is loaded, clear the image display area.

  """

  # ----------------------------
  # adata_path = f"{data_dir}/{full_sample_id}/{full_sample_id}.h5ad"
  data_dir = os.path.dirname(adata_path)
  full_sample_id = os.path.split(data_dir)[-1]
  logger.info(f'data dir: {data_dir}')
  logger.info(f'full sample id: {full_sample_id}')
  logger.info(f'loading anndata: {adata_path}')
  ad = sc.read_h5ad(adata_path)

  logger.info(f'scaling values for display')
  sc.pp.log1p(ad)

  logger.info(f'Visualizing {ad.shape} cells x variables')
  data = ad.obs.copy()
  if issparse(ad.X):
    data[ad.var_names.tolist()] = pd.DataFrame(ad.X.toarray(), index=ad.obs_names, columns=ad.var_names)
  else:
    data[ad.var_names.tolist()] = pd.DataFrame(ad.X.copy(), index=ad.obs_names, columns=ad.var_names)
  data['index_num'] = np.arange(data.shape[0])

  # coords are stored with Y inverted for plotting with matplotlib.. flip it back for pulling from the images.
  # if 'coordinates_shift' in ad.obsm.keys():
  #   logger.info(f'Using coordinates_shift as scatter coordinates')
  #   coords = ad.obsm['coordinates_shift']
  # else:
  #   logger.info(f'Using coordinates as scatter coordinates')
  #   coords = ad.obsm['coordinates']

  coords = ad.obsm['coordinates']
  data['coordinates_1'] = coords[:,0].copy()
  data['coordinates_2'] = coords[:,1].copy()
  coords[:,1] = -coords[:,1]

  # n_clusters = len(np.unique(ad.obs[annotation_col]))
  # if f'{annotation_col}_colors' in ad.uns.keys():
  #   cluster_colors = ad.uns[f'{annotation_col}_colors']
  #   color_map = {k: v for k, v in zip(np.unique(ad.obs[annotation_col]), cluster_colors)}
  # else:
  #   cluster_colors = sns.color_palette('Set1', n_clusters)
  #   cluster_colors = np.concatenate([cluster_colors, np.ones((n_clusters, 1))], axis=1)
  #   color_map = {k: rgb2hex(v) for k, v in zip(np.unique(ad.obs[annotation_col]), cluster_colors)}
  # data['color'] = [color_map[g] for g in ad.obs[annotation_col]]

  data['color'] = ['#94b5eb']*ad.shape[0]


  ## -------------------------------------------------------------------
  #                      Shared variables
  ## -------------------------------------------------------------------

  # clusters = np.array(ad.obs[annotation_col])
  # all_clusters = list(np.unique(clusters))
  # all_channels = [k for k, i in ad.uns['image_sources'].items()]
  # all_channels = sorted(ad.var_names.to_list())
  all_channels = ad.uns['channels']
  active_raw_images = {c: None for c in all_channels}
  saturation_vals = {c: (0,0) for c in all_channels} # TODO config default saturation values
  neighbor_indices = get_neighbors(coords)

  image_sources = {c: get_channel_image_path(data_dir, c) for c in sorted(all_channels)}
  for k,v in image_sources.items():
    logger.info(f'{k}: {v}')

  channel_colors = np.array(sns.color_palette('Set1', n_colors=len(all_channels)))
  channel_colors = np.concatenate([channel_colors, np.ones((len(all_channels), 1))], axis=1)
  channel_colors = {c: rgb2hex(color) for c, color in zip(all_channels, channel_colors)}

  # path for nuclear segmentation
  nuclei_path = f'{data_dir}/{full_sample_id}_2_nuclei.tif'

  _shared_variables = dict(
    # clusters = clusters,
    n_cells = ad.shape[0],
    # all_clusters = all_clusters,
    all_channels = all_channels,
    bbox = [0,0,0,0],
    active_raw_images = active_raw_images,
    saturation_vals = saturation_vals,
    active_channel = 'DAPI',
    use_channels = ['DAPI'],
    channel_colors = channel_colors,
    nbins = 100,
    image_sources = image_sources,
    background_color = '#636363',
    foreground_color = '#e34a33',
    neighbor_color = '#5e6bf2',
    coords = coords,
    neighbors = neighbor_indices,
    box_selection = np.ones(coords.shape[0], dtype=bool),
    cluster_selection = np.ones(coords.shape[0], dtype=bool),
    highlight_cells = np.zeros(coords.shape[0], dtype=bool),
    neighbor_cells = np.zeros(coords.shape[0], dtype=bool),
    nuclei_path = nuclei_path,
    adata_data=data,
    color=['#94b5eb']*ad.shape[0],
    adata=ad,
    y_shift=min(-coords[:,1]),
    x_shift=min(coords[:,0]),
    categoricals = list(ad.obs.columns[ad.obs.dtypes == 'category'])
  )

  shared_variables.update(_shared_variables)

  # return shared_variables