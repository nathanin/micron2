#!/usr/bin/env python
import numpy as np
import pandas as pd
import pytiff
import h5py
import os
import warnings

from itertools import product as iter_product

try:
  import cv2
except:
  warnings.warn('Failed to import cv2')

from skimage.measure import label
from skimage.transform import downscale_local_mean
from tqdm.auto import tqdm

"""
Given a table of coordinates, and paths to image files, 
pull out all the nuclei as little windows and stash them
in an hdf5 dataset.

The data structure for this package will be: 

hdf5/
  cells/
    channels
  meta/
    channel names
    cell id 
    coordinates

Data for each channel will be stored in its own dataset
We can work on a workflow to stash concentration and imaging parameters as attributes 

coordinates are 2D spatial coordinates of the cells in situ and allow us to generate 
outputs, and to build spatial neighbor graphs directly from the hdf5

"""

def crunch_img(img):
  return (255 * (img / 2**16)).astype(np.uint8)



def get_nuclear_masks(nuclei_img, xy_coords, sizeh, write_size):
  with pytiff.Tiff(nuclei_img, "r") as f:
    img = f.pages[0][:]

  labelimg, n_labels = label(img, connectivity=1, return_num=True)

  masks = []
  for c in tqdm(xy_coords):
      x, y = c
      bbox = [y-sizeh, y+sizeh, x-sizeh, x+sizeh]
      subimg = labelimg[bbox[0]:bbox[1], bbox[2]:bbox[3]]

      l = subimg[sizeh, sizeh]
      subimg = cv2.resize(subimg, dsize=(write_size, write_size),
                          interpolation=cv2.INTER_NEAREST)

      #plt.matshow(subimg==l)
      masks.append(subimg==l)
      
  masks = np.stack(masks, axis=0)
  return masks


def get_channel_means(h5f, group_name='intensity', 
                      idkey='meta/Cell_IDs',
                      use_masks=True,
                      mask_dataset='meta/nuclear_masks',
                      return_values=False):
  """
  Use data stored in hdf5 cell image dataset to get channel means per cell.

  This function creates new datasets under the given `group_name` representing
  the mean intensities of each channel in the cells.

  Args:
    h5f (h5py.File object)
    group_name (str): Group to place the means (default: intensity)
    return_values (bool): If true, return np.arrays, if false, write to the h5f dataset (h5f must be in w or r+ mode).
  Returns:
    vals (dict): keys: channel names
  """
  n_cells = len(h5f[idkey])
  channel_names = [b.decode('UTF-8') for b in h5f['meta/channel_names'][:]]
  vals = {k: np.zeros(n_cells, dtype=np.float32) for k in channel_names}

  if use_masks:
    masks = h5f[mask_dataset][:]

  for channel in channel_names:
    data_stack = h5f[f'cells/{channel}'][:]
    pbar = tqdm(range(n_cells))
    pbar.set_description(f'Channel {channel}')
    for i in pbar:
      data = data_stack[i]

      if use_masks:
        mask = masks[i]
        data = data[mask]

      vals[channel][i] = np.mean(data)
      if i % 25000 == 0:
        pbar.set_description(f'Channel {channel} running mean: {np.mean(vals[channel]):3.4e}')
  
  for channel in channel_names:
    d = h5f.create_dataset(f'{group_name}/{channel}', data=vals[channel])
    d.attrs['description'] = f'mean intensity of {channel} channel'
    d.attrs['mean'] = np.mean(vals[channel])
    d.attrs['std'] = np.std(vals[channel])
  h5f.flush()

  if return_values:
    return vals 


def create_nuclei_dataset(coords, image_paths, h5f, size, min_area, nuclei_img, 
                          channel_names, scale_factor, debug=False):
  h0 = pytiff.Tiff(image_paths[0])
  sizeh = int(size/2)
  h, w = h0.shape
  maxh = h - sizeh
  maxw = w - sizeh
  h0.close()

  # remove coords too near the edges:
  # remember, x = "width" = size[1]; y = "height" = size[0]
  coords = coords.query("X > @sizeh & X < @maxw & Y > @sizeh & Y < @maxh")
  if min_area is not None:
    coords = coords.query("Size > @min_area")

  # ## Debug option
  if debug:
    coords = coords.iloc[:50]

  # Downsampling option
  write_size = int(np.floor(size * scale_factor))

  datasets = []
  for c in channel_names:
    d = h5f.create_dataset(f'cells/{c}', shape=(coords.shape[0],write_size,write_size), 
                           maxshape=(None,write_size,write_size),
                           chunks=(1,write_size,write_size), 
                           dtype='uint8', 
                           compression='gzip')
    datasets.append(d)

  print(f'Pulling {coords.shape[0]} cells')
  for pth, d, c in zip(image_paths, datasets, channel_names):
    h = pytiff.Tiff(pth)
    page = h.pages[0][:]
    
    i = 0
    pbar = tqdm(zip(coords.X, coords.Y))
    pbar.set_description(f'Pulling nuclei from channel {c}')
    for x, y in pbar:
      bbox = [y-sizeh, y+sizeh, x-sizeh, x+sizeh]
      img = (255 * (page[bbox[0]:bbox[1], bbox[2]:bbox[3]] / 2**16)).astype(np.uint8)

      if scale_factor != 1:
        # img = cv2.resize(img, dsize=(0,0), fx=scale_factor, fy=scale_factor)
        img = cv2.resize(img, dsize=(write_size, write_size))

      d[i,...] = img
      i += 1

    h.close()
    h5f.flush()

  # Use a separate dataset to store cell IDs from the table
  cell_ids = [f'cell_{x}_{y}' for x, y in zip(coords.X, coords.Y)]
  cell_ids = np.array(cell_ids, dtype='S')
  d = h5f.create_dataset(f'meta/Cell_IDs', data=cell_ids)

  # Image size
  # d = h5f.create_dataset('meta/img_size', data=np.array(size, dtype=int))
  d = h5f['cells']
  d.attrs['original_size'] = size
  d.attrs['written_size'] = write_size
  d.attrs['scale_factor'] = scale_factor

  # Cell coordinates 
  xy_coords = np.array(coords.loc[:, ['X', 'Y']])
  d = h5f.create_dataset('meta/cell_coordinates', data=xy_coords)

  # If a mask is provided, store individual masks for each nucleus and get 
  # channel means constrained to the area under the nuclear mask for each cell.
  if nuclei_img is not None:
    masks = get_nuclear_masks(nuclei_img, xy_coords, sizeh, write_size)
    d = h5f.create_dataset('meta/nuclear_masks', data=masks)
    get_channel_means(h5f, group_name='cell_intensity', return_values=False)





def create_image_dataset(image_paths, h5f, size, channel_names, 
                         scale_factor, overlap, min_area, debug=False):

  h0 = pytiff.Tiff(image_paths[0])
  sizeh = int(size/2)
  h, w = h0.shape
  maxh = h - sizeh
  maxw = w - sizeh
  h0.close()

  ## Generate coordinates for tiles upper-left corners, according to size and overlap settings
  # 1. Open one of the images to get the total dimensions 
  h = pytiff.Tiff(image_paths[0])
  ht, wd = h.shape[:2]
  h.close()

  # 2. generate lists of y,x coordinates upper-left corners 
  step = int(np.floor(size * (1-overlap)))
  print(f'Tiling with step size {step}')
  xx = np.arange(0, 1+wd-size, step) ## Across
  yy = np.arange(0, 1+ht-size, step) ## Up/Down

  # 3. join coordinates into a list
  coords = [(x,y) for x,y in iter_product(xx, yy)]

  # 4. scan possible coordinates for presence of nuclei
  #   - filter tiles
  #   - assign tile IDs
  #   - track nuclei encapsulated in each tiles boundary
  # Remember we already have nuclei coords stored in h5f
  cell_ids = np.array([c.decode('utf-8') for c in h5f['meta/Cell_IDs'][:]])
  nuclei_coords = pd.DataFrame(h5f['meta/cell_coordinates'][:], index=cell_ids, columns=['X', 'Y'])
  print(f'Loaded cell coordinates: {cell_ids.shape} {nuclei_coords.shape}')

  # nc = np.array(nuclei_coords.loc[:, ['X', 'Y']])
  print('Filtering tiles without nuclei...')
  print(f'Start with {len(coords)} images')
  use_coords = []
  tile_ids = []
  encapsulated_cell_ids = {} # dictionary cell IDs contained in each tile; there may be repeats.
  for c in coords:
    y, x = c # This flip here remains confusing, but important.
    b1 , b2, b3, b4 = x, x+size, y, y+size
    encapsulated_indices = nuclei_coords.query("X > @b3 & X < @b4 & Y > @b1 & Y < @b2")
    if len(encapsulated_indices) > 0:
      t_id = f'tile_{x}_{y}'
      tile_ids.append(t_id)
      encapsulated_cell_ids[t_id] = encapsulated_indices.index.tolist()
      use_coords.append(c)
  coords = use_coords
  print(f'Finished filtering with {len(coords)} remaining images')

  if debug:
    coords = coords[:50]

  # Downsampling option
  write_size = int(np.floor(size * scale_factor))

  datasets = []
  for c in channel_names:
    d = h5f.create_dataset(f'images/{c}', shape=(len(coords),write_size,write_size), 
                           maxshape=(None,write_size,write_size),
                           chunks=(1,write_size,write_size), 
                           dtype='uint8', 
                           compression='gzip')
    datasets.append(d)

  print(f'Pulling {len(coords)} images')
  for pth, d, c in zip(image_paths, datasets, channel_names):
    h = pytiff.Tiff(pth)
    page = h.pages[0][:]
    
    i = 0
    pbar = tqdm(coords)
    pbar.set_description(f'Pulling tiles from channel {c}')
    for coord in pbar:
      y, x = coord
      # bbox = [y-sizeh, y+sizeh, x-sizeh, x+sizeh]
      bbox = [x, x+size, y, y+size]
      img = (255 * (page[bbox[0]:bbox[1], bbox[2]:bbox[3]] / 2**16)).astype(np.uint8)

      if scale_factor != 1:
        # img = cv2.resize(img, dsize=(0,0), fx=scale_factor, fy=scale_factor)
        img = cv2.resize(img, dsize=(write_size, write_size))

      d[i,...] = img
      i += 1

    h.close()
    h5f.flush()


  encapsulated_cell_ids_s = str(encapsulated_cell_ids)
  h5f['images'].attrs['tile_encapsulated_cells'] = encapsulated_cell_ids_s

  # Use a separate dataset to store cell IDs from the table
  # tile_ids = [f'tile_{x}_{y}' for x, y in coords]
  tile_ids = np.array(tile_ids, dtype='S')
  d = h5f.create_dataset(f'meta/Tile_IDs', data=tile_ids)

  # for annotating the nuclei with centroids contained in each tile
  nuclei_positions = np.arange(nuclei_coords.shape[0])

  # Collect bounding boxes
  bboxes = np.zeros((len(coords), 4), dtype=np.int)
  for i, coord in enumerate(coords):
    y, x = coord
    bbox = [x, x+size, y, y+size]
    bboxes[i,:] = bbox
  d = h5f.create_dataset(f'meta/bounding_boxes', data=bboxes)
  d.attrs['description'] = 'Bounding boxes for image tiles'

  # Tile coordinates 
  xy_coords = np.array(coords).astype(np.int)
  d = h5f.create_dataset('meta/tile_coordinates', data=xy_coords)

  get_channel_means(h5f, group_name='tile_intensity', idkey='meta/Tile_IDs',
                    use_masks=False,
                    return_values=False)

  # Image size
  # d = h5f.create_dataset('meta/img_size', data=np.array(size, dtype=int))
  d = h5f['images']
  d.attrs['original_size'] = size
  d.attrs['written_size'] = write_size
  d.attrs['scale_factor'] = scale_factor




def pull_nuclei(coords, image_paths, out_file='dataset.hdf5', nuclei_img=None,
                size=64, min_area=100, scale_factor=1., tile_scale_factor=1.,
                overlap=0, tile_size=256, channel_names=None, 
                debug=False):
  """
  Build a codex image dataset

  ** NOTE this function converts image data from uint16 to uint8 by default **

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  WHAT TO DO WITH MULTIPLE SLIDES

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  Creates an hdf5 file with datasets like:
    cells/DAPI
    cells/CD45
    ...

    tiles/DAPI
    tiles/CD45
    ...

  And meta datasets:
    meta/Cell_IDs
    meta/Tile_IDs
    meta/channel_names
    meta/nuclear_masks
    meta/img_size
    meta/image_sources

  Args:
    coords (pd.DataFrame): must have columns `X`, `Y` and `Size`
    image_paths (list, tuple): paths to uint16 TIF images, one for each channel
    out_file (str): path to store the created dataset
    nuclei_img (str): path to a nuclei label image
    size (int): size of nuclei image to pull
    min_area (float): lower bound on the area of nuclei to include
    scale_factor (float): 0 < scale_factor < 1 means downsampling, >1 means upsampling the nuclei
    tile_scale_factor (float): 0 < scale_factor < 1 means downsampling, >1 means upsampling the tiles
    overlap (float): the percentage of overlap between neighboring tiles
    tile_size (int): the tile size to pull on the rectangular grid of tiles that contain nuclei
    channel_names (list): names for the channels, same order as `image_paths`. Otherwise names are created.
    debug (bool): if True, subsets to the first 50 nuclei and tiles for quick testing.
  """

  if channel_names is None:
    channel_names = [f'ch{i:02d}' for i in range(len(image_paths))]
  assert len(channel_names) == len(image_paths)
  
  ## Ready to go
  print(f'Creating hdf5 file at {out_file}')
  h5f = h5py.File(out_file, "w")
  
  # Store the source channel names 
  channel_names_ds = np.array(channel_names, dtype='S')
  d = h5f.create_dataset('meta/channel_names', data=channel_names_ds)

  image_sources_dict = {ch: pth for ch, pth in zip(channel_names, image_paths)}
  h5f['meta'].attrs['image_sources'] = str(image_sources_dict)

  create_nuclei_dataset(coords, image_paths, h5f, size, min_area, nuclei_img, 
                        channel_names, scale_factor, debug=debug)

  create_image_dataset(image_paths, h5f, tile_size, channel_names, 
                       tile_scale_factor, overlap, min_area, debug=debug)

  

  h5f.close()



if __name__ == '__main__':
  import argparse
  import glob
  parser = argparse.ArgumentParser()
  parser.add_argument('cell_file')
  parser.add_argument('image_dir')
  parser.add_argument('out_file')

  parser.add_argument('--size', type=int, default=64)
  parser.add_argument('--min_area', type=int, default=None)

  ARGS = parser.parse_args()
  cells = pd.read_csv(ARGS.cell_file, index_col=0, header=0)
  imagefs = glob.glob(f'{ARGS.image_dir}/*.tif')
  dapi_images = [f for f in imagefs if 'DAPI' in f]
  non_dapi_images = [f for f in imagefs if 'DAPI' not in f]
  non_dapi_images = [f for f in non_dapi_images if 'Blank' not in f]
  non_dapi_images = [f for f in non_dapi_images if 'Empty' not in f]

  channel_names = [os.path.basename(x) for x in non_dapi_images]
  channel_names = [x.replace(f'.tif','') for x in channel_names]
  channel_names = [x.split('_')[-2] for x in channel_names]
  # channel_names = [x.replace('-', '_') for x in channel_names]
  channel_names = ["DAPI"] + channel_names

  image_paths = [dapi_images[0]] + non_dapi_images
  #image_handles = [pytiff.Tiff(dapi_images[0])] + [pytiff.Tiff(f) for f in non_dapi_images]
  pull_nuclei(cells, image_paths, out_file=ARGS.out_file, 
              size=ARGS.size, min_area=ARGS.min_area, 
              channel_names=channel_names)
