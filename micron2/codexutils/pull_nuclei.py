#!/usr/bin/env python
import numpy as np
import pandas as pd
import pytiff
import h5py
import os

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


def pull_nuclei(coords, image_paths, out_file='dataset.hdf5', size=64, min_area=100, channel_names=None):
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
  
  # remove coords too near the edges:
  # remember, x = "width" = size[1]; y = "height" = size[0]
  coords = coords.query("X > @sizeh & X < @maxw & Y > @sizeh & Y < @maxh")
  if min_area is not None:
    coords = coords.query("Size > @min_area")

  # ## Debug option
  # coords = coords.iloc[:5000]

  datasets = []
  for c in channel_names:
    d = h5f.create_dataset(f'cells/{c}', shape=(coords.shape[0],size,size), maxshape=(None,size,size),
                            dtype='uint8', chunks=(1,size,size), compression='gzip')
    datasets.append(d)
      
  
  print(f'Pulling {coords.shape[0]} cells')
  for pth, d, c in zip(image_paths, datasets, channel_names):
    h = pytiff.Tiff(pth)
    page = h.pages[0][:]
    
    i = 0
    pbar = tqdm(zip(coords.X, coords.Y))
    pbar.set_description(f'Pulling from channel {c}')
    for x, y in pbar:
      bbox = [y-sizeh, y+sizeh, x-sizeh, x+sizeh]
      img = (255 * (page[bbox[0]:bbox[1], bbox[2]:bbox[3]] / 2**16)).astype(np.uint8)
      d[i,...] = img
      i += 1

    h.close()
    h5f.flush()

  # Use a separate dataset to store cell IDs from the table
  cell_ids = np.array(coords.index, dtype='S')
  d = h5f.create_dataset(f'meta/Cell_IDs', data=cell_ids)

  # Store information like the source channel names 
  channel_names = np.array(channel_names, dtype='S')
  d = h5f.create_dataset('meta/channel_names', data=channel_names)

  # In-situ coordinates 
  channel_names = np.array(channel_names, dtype='S')
  d = h5f.create_dataset('meta/coordinates', data=np.array(coords.loc[:, ['X', 'Y']]))

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
