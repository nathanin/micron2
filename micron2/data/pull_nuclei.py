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
from skimage.filters import threshold_otsu
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



def get_masks(nuclei_img, xy_coords, sizeh, write_size):
  with pytiff.Tiff(nuclei_img, "r") as f:
    img = f.pages[0][:]

  labelimg, n_labels = label(img, connectivity=1, return_num=True)

  masks = []
  for c in tqdm(xy_coords, total=len(xy_coords), disable=None):
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
    with tqdm(range(n_cells), total=n_cells, disable=None) as pbar:
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


def image_stats(pixels):
  mean = np.mean(pixels)
  pct = np.mean(pixels > 0)
  sd = np.std(pixels)
  qs = np.quantile(pixels, [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
  pix_log = np.log1p(pixels)
  npix = np.prod(pixels.shape)
  c, _ = np.histogram(pix_log, bins=8)
  c = c / npix

  info = np.array([ mean, sd, pct] + list(qs) + list(c))
  info_labels = 'mean,std,percent_positive,q01,q10,q25,q50,q75,q90,q95,q99,b1,b2,b3,b4,b5,b6,b7,b8'
  return info, info_labels


def create_nuclei_dataset(coords, image_paths, h5f, size, min_area, nuclei_img, membrane_img, tissue_img,
                          channel_names, scale_factor, cell_prefix='cell', min_thresh=15, debug=False):
  """ Pull raw data from the images provided according to coordinate locations
  """
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
  mean_datasets = []
  for c in channel_names:
    d = h5f.create_dataset(f'cells/{c}', shape=(coords.shape[0],write_size,write_size), 
                           maxshape=(None,write_size,write_size),
                           chunks=(1,write_size,write_size), 
                           dtype='uint8', 
                           compression='gzip')
    datasets.append(d)

  # Cell coordinates 
  xy_coords = np.array(coords.loc[:, ['X', 'Y']])
  d = h5f.create_dataset('meta/cell_coordinates', data=xy_coords)

  # If a mask is provided, store individual masks for each nucleus and get 
  # channel means constrained to the area under the nuclear mask for each cell.
  if nuclei_img is not None:
    masks = get_masks(nuclei_img, xy_coords, sizeh, write_size)
    d = h5f.create_dataset('meta/nuclear_masks', data=masks)
    # get_channel_means(h5f, group_name='cell_intensity', mask_dataset='meta/nuclear_masks', return_values=False)

  if membrane_img is not None:
    masks = get_masks(membrane_img, xy_coords, sizeh, write_size)
    d = h5f.create_dataset('meta/membrane_masks', data=masks)
    # get_channel_means(h5f, group_name='membrane_intensity', mask_dataset='meta/membrane_masks', return_values=False)

  if tissue_img is not None:
    tissue = cv2.imread(tissue_img,-1) > 0

  # Commence pulling cells 
  print(f'Pulling {coords.shape[0]} cells')
  for pth, d, c in zip(image_paths, datasets, channel_names):
    h = pytiff.Tiff(pth)
    page = h.pages[0][:]

    # Set up for pulling stats
    # if nuclei_img is not None:
    nuclei_stats = []
    membrane_stats = []

    # # Use the FIJI/ImageJ 'Auto' Contrast histogram method to find a low cutoff
    # if tissue_img is not None:
    #   N, bins = np.histogram(page[tissue].ravel(), 256)
    # else:
    #   N, bins = np.histogram(page.ravel(), 256)

    # npix = np.cumsum(N)
    # target = np.prod(page.shape)/10
    # bin_index = np.argwhere(npix > target)[0,0]
    # thr = int(bins[bin_index+1])
    # thr = max(thr, min_thresh)
    thr = 0
    high_sat = np.quantile(page, 0.999)
    d.attrs['threshold'] = thr

    i = 0
    with tqdm(zip(coords.X, coords.Y), total=coords.shape[0], disable=None) as pbar:
      pbar.set_description(f'Pulling nuclei from channel {c}')
      for x, y in pbar:
        bbox = [y-sizeh, y+sizeh, x-sizeh, x+sizeh]
        img_raw = page[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        
        ## Block commented --- no modification of the original image
        # # do not alter the DAPI
        # if c=='DAPI':
        #   pass
        # else:
        #   thr_mask = img_raw<thr
        #   img_raw[thr_mask] = 0
        #   img_raw[~thr_mask] = img_raw[~thr_mask]-thr

        nuclei_mask = h5f['meta/nuclear_masks'][i,...]
        img_info, info_labels = image_stats(img_raw[nuclei_mask].ravel())
        nuclei_stats.append(img_info.copy())

        membrane_mask = h5f['meta/membrane_masks'][i,...]
        img_info, info_labels = image_stats(img_raw[membrane_mask].ravel())
        membrane_stats.append(img_info.copy())

        ## Adjust low values and convert to uint8...
        img = np.ceil(255 * (img_raw / high_sat)).astype(np.uint8)
        # img = img_raw

        if scale_factor != 1:
          img = cv2.resize(img, dsize=(write_size, write_size))

        d[i,...] = img
        i += 1

    nuclei_stats = np.stack(nuclei_stats, axis=0).astype(np.float32)
    print(f'Channel {c} got {100*(nuclei_stats[:,0]==0).mean():3.3f}% nuclei zeros')

    nuclei_stat_dataset = h5f.create_dataset(f'cell_nuclei_stats/{c}', data=nuclei_stats)
    nuclei_stat_dataset.attrs['label'] = info_labels#'mean,std,percent_positive,q01,q10,q25,q50,q75,q90,q95,q99'
    nuclei_stat_dataset.attrs['mean'] = np.mean(nuclei_stats, axis=0)
    nuclei_stat_dataset.attrs['std'] = np.std(nuclei_stats, axis=0)

    membrane_stats = np.stack(membrane_stats, axis=0).astype(np.float32)
    print(f'Channel {c} got {100*(membrane_stats[:,0]==0).mean():3.3f}% membrane zeros')

    membrane_stat_dataset = h5f.create_dataset(f'cell_membrane_stats/{c}', data=membrane_stats)
    membrane_stat_dataset.attrs['label'] = 'mean,std,percent_positive,q01,q10,q25,q50,q75,q90,q95,q99'
    membrane_stat_dataset.attrs['mean'] = np.mean(membrane_stats, axis=0)
    membrane_stat_dataset.attrs['std'] = np.std(membrane_stats, axis=0)

    h.close()
    h5f.flush()

  # Use a separate dataset to store cell IDs from the table
  cell_ids = [f'{cell_prefix}_{x}_{y}' for x, y in zip(coords.X, coords.Y)]
  cell_ids = np.array(cell_ids, dtype='S')
  d = h5f.create_dataset(f'meta/Cell_IDs', data=cell_ids)

  # Image size
  # d = h5f.create_dataset('meta/img_size', data=np.array(size, dtype=int))
  d = h5f['cells']
  d.attrs['original_size'] = size
  d.attrs['written_size'] = write_size
  d.attrs['scale_factor'] = scale_factor





def create_image_dataset(image_paths, h5f, size, channel_names, 
                         scale_factor, overlap, tile_prefix='tile', 
                         tissue_img=None,
                         min_thresh=15, debug=False):

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
      t_id = f'{tile_prefix}_{x}_{y}'
      tile_ids.append(t_id)
      encapsulated_cell_ids[t_id] = encapsulated_indices.index.tolist()
      use_coords.append(c)
  coords = use_coords
  print(f'Finished filtering with {len(coords)} remaining images')

  if tissue_img is not None:
    tissue = cv2.imread(tissue_img, -1) > 0
 
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

    # Use the FIJI/ImageJ 'Auto' Contrast histogram method to find a low cutoff
    if tissue_img is not None:
      N, bins = np.histogram(page[tissue].ravel(), 256)
    else:
      N, bins = np.histogram(page.ravel(), 256)

    # npix = np.cumsum(N)
    # target = np.prod(page.shape)/10
    # bin_index = np.argwhere(npix > target)[0,0]
    # thr = int(bins[bin_index+1])
    # thr = max(min_thresh, thr)
    thr = 0
    d.attrs['threshold'] = thr

    # if 'DAPI' in c:
    #   print('Skipping DAPI channel thresholding')
    #   pass
    # else:
    #   print(f'Channel {c} subtracting constant {thr}')
    
    i = 0
    channel_stats = []
    with tqdm(coords, total=len(coords), disable=None) as pbar:
      pbar.set_description(f'Pulling tiles from channel {c}')
      for coord in pbar:
        y, x = coord
        # bbox = [y-sizeh, y+sizeh, x-sizeh, x+sizeh]
        bbox = [x, x+size, y, y+size]
        raw_img = page[bbox[0]:bbox[1], bbox[2]:bbox[3]]

        # if 'DAPI' in c:
        #   pass
        # else:
        #   thr_mask = raw_img < thr
        #   raw_img[thr_mask] = 0
        #   raw_img[~thr_mask] = raw_img[~thr_mask] - thr

        # img_avg = np.mean(raw_img)
        img_info, info_labels = image_stats(raw_img.ravel())
        channel_stats.append(img_info.copy())

        # raw_img[raw_img<thr] = 0
        img = np.ceil(255 * (raw_img / 2**16)).astype(np.uint8)

        if scale_factor != 1:
          # img = cv2.resize(img, dsize=(0,0), fx=scale_factor, fy=scale_factor)
          img = cv2.resize(img, dsize=(write_size, write_size))

        d[i,...] = img
        i += 1

    channel_stats = np.stack(channel_stats, axis=0).astype(np.float32)
    stats_dataset = h5f.create_dataset(f'tile_stats/{c}', data=channel_stats)
    stats_dataset.attrs['label'] = info_labels #'mean,std,percent_positive,q01,q10,q25,q50,q75,q90,q95,q99'
    stats_dataset.attrs['mean'] = np.mean(channel_stats)
    stats_dataset.attrs['std'] = np.std(channel_stats)

    h.close()
    h5f.flush()

  # for annotating the nuclei with centroids contained in each tile
  encapsulated_cell_ids_s = str(encapsulated_cell_ids)
  h5f['images'].attrs['tile_encapsulated_cells'] = encapsulated_cell_ids_s

  # Use a separate dataset to store cell IDs from the table
  tile_ids = np.array(tile_ids, dtype='S')
  d = h5f.create_dataset(f'meta/Tile_IDs', data=tile_ids)

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

  # get_channel_means(h5f, group_name='tile_intensity', idkey='meta/Tile_IDs',
  #                   use_masks=False,
  #                   return_values=False)

  # Image size
  # d = h5f.create_dataset('meta/img_size', data=np.array(size, dtype=int))
  d = h5f['images']
  d.attrs['original_size'] = size
  d.attrs['written_size']  = write_size
  d.attrs['scale_factor']  = scale_factor




def pull_nuclei(coords, image_paths, out_file='dataset.hdf5', nuclei_img=None,
                membrane_img=None, tissue_img=None, size=64, min_area=100, scale_factor=1., tile_scale_factor=1.,
                overlap=0, tile_size=128, channel_names=None, 
                skip_tiles=False,
                debug=False):
  """
  Build a codex image dataset

  ** NOTE this function converts image data from uint16 to uint8 by default **

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
    nuclei_img (str): path to a nuclei label image or binary mask
    membrane_img (str): path to a membrane label image or binary mask
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

  # TODO hook in command line option to prefix cells with e.g. sample name
  create_nuclei_dataset(coords, image_paths, h5f, size, min_area, nuclei_img, membrane_img, tissue_img,
                        channel_names, scale_factor, cell_prefix='cell', debug=debug)

  if not skip_tiles:
    create_image_dataset(image_paths, h5f, tile_size, channel_names,
                        tile_scale_factor, overlap, min_area, tissue_img=tissue_img, debug=debug)

  h5f.close()
