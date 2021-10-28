import scanpy as sc
import pandas as pd
import numpy as np

import glob
import os
import itertools

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


import json
def load_celltype_config(config_path, shared_variables, logger):
  logger.info(f'Loading celltype channel defaults from {config_path}')

  celltype_channels = json.load(open(config_path, 'r'))
  for k,v in celltype_channels.items():
    logger.info(f'loaded celltype channels {k}: {v}')


  celltype_choices = list(celltype_channels.keys())

  colors = sns.color_palette('tab20', n_colors=len(celltype_choices))
  celltype_colors = {}
  for c,clr in zip(celltype_choices, colors): 
    celltype_colors[c] = rgb2hex(clr)

  _shared_variables = dict(
    celltype_channels = celltype_channels,
    celltype_choices = celltype_choices,
    celltype_colors = celltype_colors
  )
  shared_variables.update(_shared_variables)

  logger.info('Returning from celltype config with shared variables:')
  for k,v in shared_variables.items():
    logger.info(f'{k}')


def load_color_config(config_path, shared_variables, logger):
  color_config = pd.read_csv(config_path, index_col=0, header=0)
  all_channels = color_config.index.tolist()

  data_dir = shared_variables['data_dir']
  image_sources = {c: get_channel_image_path(data_dir, c) for c in sorted(all_channels)}
  for k,v in image_sources.items():
    logger.info(f'{k}: {v}')

  saturation_vals = {c: (int(color_config.loc[c,'low']), int(color_config.loc[c,'high'])) for c in all_channels}
  channel_colors = {c: color_config.loc[c, 'color'] for c in all_channels}

  _shared_variables = dict(
    color_config = color_config,
    all_channels = all_channels,
    saturation_vals = saturation_vals,
    channel_colors = channel_colors,
    image_sources = image_sources,
    color_wheel = ['a2a2a2', '#8823c2', '#3fdb2e', '#c40e0e', '#d18b08']
  )
  shared_variables.update(_shared_variables)
  logger.info('Returning from color config with shared variables:')
  for k,v in shared_variables.items():
    logger.info(f'{k}')


def find_populated_regions(coords, tilesize=3096, min_cells=500):
  max_row, max_col = np.max(coords, axis=0)

  nrow = int(np.floor(max_row / tilesize))
  rows = np.linspace(0, max_row-tilesize, nrow, dtype=int)

  ncol = int(np.floor(max_col / tilesize))
  cols = np.linspace(0, max_col-tilesize, ncol, dtype=int)

  use_tiles = []
  for c,r in itertools.product(cols, rows):
    cells_row = (coords[:,0] > r) & (coords[:,0] < r+tilesize)
    cells_col = (coords[:,1] > c) & (coords[:,1] < c+tilesize)
    tile_cells = cells_col & cells_row
    bbox = (r,r+tilesize,c,c+tilesize)
    if np.sum(tile_cells) > min_cells:
      use_tiles.append(bbox)

  return use_tiles


def _generate_regions(coords):
  for co in itertools.cycle(coords):
    # yield (co[0], co[0]+tilesize, co[1], co[1]+tilesize)
    yield co


def set_active_slide(data_dir, shared_variables, logger):

  # data_dir = os.path.dirname(csv_path)
  data_dir = data_dir[:-1] if data_dir.endswith('/') else data_dir

  full_sample_id = os.path.split(data_dir)[-1]
  csv_path = f'{data_dir}/{full_sample_id}_2_centroids.csv'
  logger.info(f'data dir: {data_dir}')
  logger.info(f'full sample id: {full_sample_id}')
  logger.info(f'loading cells: {csv_path}')

  annotation_path = f'{data_dir}/{full_sample_id}_3_annotation.csv'
  if os.path.exists(annotation_path):
    logger.info(f'loading annotations from {annotation_path}')
    cells = pd.read_csv(annotation_path, index_col=0, header=0, 
      dtype={'celltype': str, 'subtype': str, 'district': str, 
             'TLS_district': str, 'X': int, 'Y': int})
    existing_annotations = ['']+[c for c in cells.columns if c not in ['X', 'Y']]
  else:
    cells = pd.read_csv(csv_path, index_col=0, header=0)
    existing_annotations = ['']

  # pad dummy coords so we can view datasets without doing segmentation first
  if cells.shape[0] == 0:
    logger.info(f'Preparing to show sample without pre-segmented cells.')
    for i in range(5):
      cells.loc[i,:] = 1

    print(cells)


  logger.info(f'Visualizing {cells.shape[0]} cells')
  cells['color'] = ['#94b5eb']*cells.shape[0]
  cells['annotation'] = [''] * cells.shape[0]
  cells['prediction'] = [''] * cells.shape[0]

  coords = np.array(cells.loc[:, ['X', 'Y']].values).copy()

  ## -------------------------------------------------------------------
  #                      Shared variables
  ## -------------------------------------------------------------------

  # path for nuclear segmentation
  nuclei_path = f'{data_dir}/{full_sample_id}_2_nuclei.tif'

  # use_tiles = find_populated_regions(coords, tilesize=1024, min_cells=200)
  # logger.info(f'Found {len(use_tiles)} densely populated tiles')

  # bbox_generator = _generate_regions(use_tiles)

  _shared_variables = dict(
    n_cells = cells.shape[0],
    data_dir = data_dir,
    cell_names = np.array(cells.index),
    nbins = 100,
    # use_tiles = use_tiles,
    # bbox_generator = bbox_generator,
    use_tiles = None,
    bbox_generator = None,
    background_color = '#636363',
    foreground_color = '#636363',
    neighbor_color = '#5e6bf2',
    default_channel_color = '#a2a2a2',
    coords = coords,
    nuclei_path = nuclei_path,
    default_dot_color = '#3d3d3d',
    cell_data = cells,
    y_shift=min(coords[:,1]) if coords.shape[0]>0 else 0,
    x_shift=min(coords[:,0]) if coords.shape[0]>0 else 0,
    existing_annotations=existing_annotations,
  )

  shared_variables.update(_shared_variables)

  logger.info('Returning from set active slide with shared variables:')
  for k,v in shared_variables.items():
    logger.info(f'{k}')
