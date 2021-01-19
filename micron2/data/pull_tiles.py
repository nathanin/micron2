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


def _get_channel_means(h5f, group_name='intensity', return_values=False):
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
  n_cells = len(h5f['meta/Tile_IDs'])
  channel_names = [b.decode('UTF-8') for b in h5f['meta/channel_names'][:]]
  vals = {k: np.zeros(n_cells, dtype=np.float32) for k in channel_names}

  for channel in channel_names:
    data_stack = h5f[f'images/{channel}'][:]
    pbar = tqdm(range(n_cells))
    pbar.set_description(f'Channel {channel}')
    for i in pbar:
      data = data_stack[i]
      vals[channel][i] = np.mean(data)
      if i % 50000 == 0:
        pbar.set_description(f'Channel {channel} running mean: {np.mean(vals[channel]):3.4e}')
  
  for channel in channel_names:
    d = h5f.create_dataset(f'{group_name}/{channel}', data=vals[channel])
    d.attrs['description'] = f'mean intensity of {channel} channel'
    d.attrs['mean'] = np.mean(vals[channel])
    d.attrs['std'] = np.std(vals[channel])
  h5f.flush()

  if return_values:
    return vals 


def pull_tiles(image_paths, out_file='dataset.hdf5', 
               nuclei_coords=None, nuclei_img=None,
               size=256, scale_factor=1., overlap=0, 
               channel_names=None, 
               debug=False):
  """
  Build a tile dataset by tiling the whole input image

  For each tile record (absolute and relative) positions of contained nuclei
  Some nuclei will be split by this function, and if overlap > 0, there 
  may be repetition of the nuclei in each tile

  Use nuclei coordinates to filter tiles

  ** NOTE this function converts from uint16 to uint8 **

  Creates an hdf5 file with datasets like:
  cells/DAPI
  cells/CD45

  And non-imaging meta datasets:
  meta/Cell_IDs
  meta/channel_names
  meta/nuclear_masks
  meta/img_size

  Args:
    image_paths (list): paths to the image data
    out_file (str): destination for the collected dataset
    nuclei_coords (pd.DataFrame): coordinates (and unique identifier) of nuclei
    nuclei_img (str): path to the nucleus image
    size (int): size in pixels of the tiles
    scale_factor (float) [> 0]: scale factor, as used in cv2.resize() 
    overlap (float) [0, 1]: percentage overlap between tiles
    channel_names (list): channels to use, if not all channels need to be saved
  """

  h0 = pytiff.Tiff(image_paths[0])
  sizeh = int(size/2)
  h, w = h0.shape
  maxh = h - sizeh
  maxw = w - sizeh
  h0.close()

  if channel_names is None:
    channel_names = [f'ch{i:02d}' for i in range(len(image_paths))]
  assert len(channel_names) == len(image_paths)
  
  print(f'Creating hdf5 file at {out_file}')
  h5f = h5py.File(out_file, "w")
  
  ## Generate coordinates for tiles upper-left corners, according to size and overlap settings
  # 1. Open one of the images to get the total dimensions 
  h = pytiff.Tiff(image_paths[0])
  ht, wd = h.shape[:2]
  h.close()

  print(f'Working from source image: {ht} x {wd}')

  # 2. generate lists of y,x coordinates upper-left corners 
  step = int(np.floor(size * (1-overlap)))
  print(f'Tiling with step size {step}')
  xx = np.arange(0, 1+wd-size, step) ## Across
  yy = np.arange(0, 1+ht-size, step) ## Up/Down

  # 3. join coordinates into a list
  coords = [(y,x) for x,y in iter_product(xx, yy)]

  # ## Debug option -- subset coordinates to go fast
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
    pbar.set_description(f'Pulling from channel {c}')
    for coord in pbar:
      x, y = coord
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

  # Collect bounding boxes
  bboxes = np.zeros((len(coords), 4), dtype=np.int)
  for i, coord in enumerate(coords):
    x, y = coord
    bbox = [x, x+size, y, y+size]
    bboxes[i,:] = bbox
  d = h5f.create_dataset(f'meta/bounding_boxes', data=bboxes)

  # Use a separate dataset to store tile IDs
  tile_ids = [f'tile_{x}_{y}' for x,y in coords]
  tile_ids = np.array(tile_ids, dtype='S')
  d = h5f.create_dataset(f'meta/Tile_IDs', data=tile_ids)

  # Store information like the source channel names 
  channel_names = np.array(channel_names, dtype='S')
  d = h5f.create_dataset('meta/channel_names', data=channel_names)

  # In-situ coordinates 
  # xy_coords = np.array(coords.loc[:, ['X', 'Y']])
  xy_coords = np.array(coords, dtype=np.int)
  d = h5f.create_dataset('meta/coordinates', data=xy_coords)

  # Image size
  # d = h5f.create_dataset('meta/img_size', data=np.array(size, dtype=int))
  d = h5f['images']
  d.attrs['original_size'] = size
  d.attrs['written_size'] = write_size
  d.attrs['scale_factor'] = scale_factor

  # If a mask is provided, store individual masks for each nucleus
  # if nuclei_img is not None:
    # masks = get_nuclear_masks(nuclei_img, xy_coords, sizeh, write_size)
    # d = h5f.create_dataset('meta/nuclear_masks', data=masks)

  _get_channel_means(h5f, group_name='intensity', return_values=False)

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
