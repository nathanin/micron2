import numpy as np
import pandas as pd
import h5py
import os

import random

import tqdm.auto as tqdm

""" Data/dataset structures and methods for working with them 

"""

def hdf5_info(h5f, show_datasets=None, show_names=True):
  """ Print information about a dataset

  Args:
    h5f (str): A path

  Returns:
    None
  """

  base = os.path.splitext(os.path.basename(h5f))[0]

  with h5py.File(h5f, 'r') as f:
    datasets = list(f.keys())
    if show_datasets is not None:
      datasets = [d for d in datasets if d in show_datasets]

    for d in datasets:
      print()
      attrs = list(f[d].attrs.keys())
      print(f'{base} {d}\t--- {len(attrs)} attributes ---')
      if show_names: 
        for a in attrs:
          print(f'{base} {d}\t{a}')

      try:
        dkeys = list(f[d].keys())
      except:
        print(f'{base} {d}\tno keys')
        continue

      print(f'{base} {d}\t--- {len(dkeys)} entries ---')
      if show_names:
        for k in dkeys:
          shape = f[f'{d}/{k}'].shape
          print(f'{base} {d}\t{k} {shape}')


def _get_hdf5_keys(h5f, dataset):
  """ List the keys associated with the provided dataset

  Args:
    h5f (str): path to hdf5 object on disk
    dataset (str): a dataset or group that exists in h5f

  Returns:
    channels (list): the keys of h5f[dataset]
  """

  with h5py.File(h5f, 'r') as f:
    ch = list(f[dataset].keys())
  return ch



def _get_size_attr(h5f, dataset):
  """ Extract the size attribute from a dataset/group """
  with h5py.File(h5f, 'r') as f:
    size = f[dataset].attrs['written_size']
  return size

def _check_concat_inputs(h5fs):
  # Check sameness of the channels
  channels = _get_hdf5_keys(h5fs[0], 'cells')
  for f in h5fs[1:]:
    channels2 = _get_hdf5_keys(f, 'cells')
    assert channels == channels2, f'Dataset {f} channel mismatch'

  # Check sameness of image sizes
  cell_size = _get_size_attr(h5fs[0], 'cells')
  for f in h5fs[1:]:
    cell_size2 = _get_size_attr(f, 'cells')
    assert cell_size == cell_size2, f'Dataset {f} cell image size mismatch ({cell_size} vs {cell_size2})'

  image_size = _get_size_attr(h5fs[0], 'images')
  for f in h5fs[1:]:
    image_size2 = _get_size_attr(f, 'images')
    assert image_size == image_size2, f'Dataset {f} cell image size mismatch ({image_size} vs {image_size2})'

def _merge_image_datasets_v2(h5fout, h5fs, dataset, channel, size, shuffle=True, seed=999):
  """ Merge all image datasets of the same name

  Args:
    h5fout (h5py.File): The open output file to be written
    h5fs (list): list of paths
    dataset (str): dataset to join
  
  Returns:
    Nothing
  """
  total_size = 0
  has_dataset = []
  for h5f in h5fs:
    with h5py.File(h5f,'r') as f:
      try:
        total_size += f[f'{dataset}/{channel}'].shape[0]
        has_dataset.append(True)
      except:
        has_dataset.append(False)

  if not all(has_dataset):
    print(f'Error merging dataset {dataset}/{channel}. Skipping...')
    return None

  print(f'Merging {dataset}/{channel} with {total_size} total elements from {len(h5fs)} files')
  d = h5fout.create_dataset(f'{dataset}/{channel}', size=(total_size, size, size),
                            compression='gzip', chunks=(1,size,size))

  offset = 0
  for h5f in h5fs:
    print(f'\tsource: {h5f}')
    with h5py.File(h5f, 'r') as f:
      data = f[f'{dataset}/{channel}'][:]
      size = data.shape[0]
      d[offset:offset+size] = data
      offset += size

  if shuffle:
    print('Shuffling')
    random.seed(seed)
    random.shuffle(d)

  print('Calculating attributes')
  d.attrs['max'] = np.max(d)
  d.attrs['mean'] = np.mean(d)
  d.attrs['std'] = np.std(d)


def _merge_image_datasets(h5fout, h5fs, dataset, channel, size, perm=None):
  """ Merge all image datasets of the same name

  Args:
    h5fout (h5py.File): The open output file to be written
    h5fs (list): list of paths
    dataset (str): dataset to join
  
  Returns:
    Nothing
  """
  ds = f'{dataset}/{channel}'
  if ds in h5fout.keys():
    print(f'Found dataset {ds}')
    return

  total_size = 0
  has_dataset = []
  for h5f in h5fs:
    with h5py.File(h5f,'r') as f:
      try:
        total_size += f[f'{dataset}/{channel}'].shape[0]
        has_dataset.append(True)
      except:
        has_dataset.append(False)

  if not all(has_dataset):
    print(f'Error merging dataset {dataset}/{channel}. Skipping...')
    return None
  
  print(f'Merging {dataset}/{channel} with {total_size} total elements from {len(h5fs)} files')

  data = []
  for h5f in h5fs:
    print(f'\tsource: {h5f}')
    with h5py.File(h5f, 'r') as f:
      data.append(f[f'{dataset}/{channel}'][:])
  data = np.concatenate(data, axis=0)
  print(f'{data.shape}')

  if perm is not None:
    data = data[perm]

  d = h5fout.create_dataset(f'{dataset}/{channel}', data=data, compression='gzip', chunks=(1,size,size))
  d.attrs['max'] = np.max(data)
  d.attrs['mean'] = np.mean(data)
  d.attrs['std'] = np.std(data)

  del data

  # d = h5fout.create_dataset(f'{dataset}/{channel}', shape=(total_size,size,size), 
  #                           maxshape=(None,size,size),
  #                           chunks=(1,size,size), 
  #                           dtype='uint8', # TODO inherit dtype from the sources
  #                           compression='gzip')
  # i=0
  # for h5f in h5fs:
  #   print(f'\tsource: {h5f}')
  #   with h5py.File(h5f,'r') as f:
  #     n=f[f'{dataset}/{channel}'].shape[0]
  #     d[i:i+n,...]=f[f'{dataset}/{channel}'][:]
  #   i=n
  # if perm is not None:
  #   d[:] = d[perm,...]

def _merge_value_datasets_v2(h5fout, h5fs, dataset, stats, shuffle=True, seed=999,
                             transfer_attr=None):
  """ Merge scalar value datasets """
  values = []
  for h5f in h5fs:
    with h5py.File(h5f,'r') as f:
      values.append(f[dataset][:])
  values = np.concatenate(values)
  print(f'Merging scalar dataset: {dataset} new shape {values.shape}')

  d = h5fout.create_dataset(dataset, data=values)
  if (transfer_attr is not None) and isinstance(transfer_attr, list):
    for a in transfer_attr:
      with h5py.File(h5fs[0],'r') as f:
        d.attr[a] = f[dataset].attr[a]
  
  if shuffle:
    print('Shuffling')
    random.seed(seed)
    random.shuffle(d)

  if stats:
    d.attrs['mean'] = np.mean(values)
    d.attrs['std'] = np.std(values)
    d.attrs['max'] = np.max(values)

def _merge_value_datasets(h5fout, h5fs, dataset, stats, perm=None, transfer_attr=None):
  """ Merge scalar value datasets """

  if dataset in h5fout.keys():
    print(f'Found dataset {dataset}')
    return

  values = []
  for h5f in h5fs:
    with h5py.File(h5f,'r') as f:
      values.append(f[dataset][:])
  values = np.concatenate(values)
  print(f'Merging scalar dataset: {dataset} new shape {values.shape}')

  if perm is not None:
    values = values[perm]

  d = h5fout.create_dataset(dataset, data=values)
  if (transfer_attr is not None) and isinstance(transfer_attr, list):
    for a in transfer_attr:
      with h5py.File(h5fs[0],'r') as f:
        d.attrs[a] = f[dataset].attrs[a]

  if stats:
    d.attrs['mean'] = np.mean(values)
    d.attrs['std'] = np.std(values)
    d.attrs['max'] = np.max(values)




def _merge_coordinates_v2(h5fout, h5fs, dataset, shuffle=True, seed=999):
  coords = []
  for h5f in h5fs:
    with h5py.File(h5f, 'r') as f:
      coords.append(f[dataset][:])
  coords = np.concatenate(coords, axis=0)

  d = h5fout.create_dataset(dataset, data=coords)
  if shuffle:
    random.seed(seed)
    random.shuffle(d)


def _merge_coordinates(h5fout, h5fs, dataset, perm=None):
  if dataset in h5fout.keys():
    print(f'Found dataset {dataset}')
    return

  coords = []
  for h5f in h5fs:
    with h5py.File(h5f, 'r') as f:
      coords.append(f[dataset][:])
  coords = np.concatenate(coords, axis=0)

  if perm is not None:
    coords = coords[perm,...]

  d = h5fout.create_dataset(dataset, data=coords)



def _make_source_dataset_v2(h5fout, h5fs, samples, size_ref_dataset, ds_name, 
                            shuffle=True, seed=999):
  """ Make a string dataset that lists the source sample for each cell/image """
  print('Make source dataset')
  values = []
  for h5f, name in zip(h5fs, samples):
    with h5py.File(h5f,'r') as f:
      size = f[size_ref_dataset].shape[0]
    values += [name]*size

  values = np.array(values, dtype='S')
  # values = values[perm]
  d = h5fout.create_dataset(f'meta/{ds_name}', data=values)
  if shuffle:
    print('Shuffing...')
    random.seed(seed)
    random.shuffle(d)


def _make_source_dataset(h5fout, h5fs, samples, size_ref_dataset, ds_name, seed=999):
  """ Make a string dataset that lists the source sample for each cell/image """
  ds = f'meta/{ds_name}'
  if ds in h5fout.keys():
    print(f'Found dataset {ds}')
    return

  values = []
  for h5f, name in zip(h5fs, samples):
    with h5py.File(h5f,'r') as f:
      size = f[size_ref_dataset].shape[0]
    values += [name]*size

  np.random.seed(seed)
  perm = np.random.choice(len(values), len(values), replace=False)
  values = np.array(values, dtype='S')
  values = values[perm]
  _ = h5fout.create_dataset(f'meta/{ds_name}', data=values)
  _ = h5fout.create_dataset(f'meta/{ds_name}_perm', data=perm)


def shift_coordinates_v2(labels, coords_in, sample_layout):
  # rows and columns are flipped again
  
  nr,nc = sample_layout.shape
  print('layout', nr,nc)
  
  coords = coords_in.copy()
  # coords = adata.obsm['coordinates'].copy()
  # coords[:,1] = -coords[:,1]
  #coords_tmp = coords.copy()
  
  row_shift_grid = np.zeros_like(sample_layout, dtype=np.int)
  col_shift_grid = np.zeros_like(sample_layout, dtype=np.int)
      
  # Cycle through once and find the shifts
  print('ranges', nr, nc)
  for r2 in range(nr):
    print('row', r2)
    if r2>0:
      ref_row = r2-1 if r2>0 else r2
      print('\treference row:', ref_row)
      row_shift = current_row_max
      print('\trow shift:', row_shift)
    else: 
      row_shift = 0
    
    current_row_max = 0
    for c2 in range(nc):
      print('\tcolumn', c2)
      if r2==0 and c2==0: continue
      
      print('\t\tlocation', r2, c2)
      target_slide = sample_layout[r2,c2]
      if target_slide == 'pass':
        continue
      if target_slide is None:
        continue
      print('\t\ttarget slide:', target_slide)
      target_idx = labels==target_slide
      target_coords = coords[target_idx].copy()
      print('\t\tstart:', max(target_coords[:,0]), max(target_coords[:,1]))
      dx = max(target_coords[:,0])-min(target_coords[:,0])
      dy = max(target_coords[:,1])-min(target_coords[:,1])
      print('\t\tdimensions:', dx, dy)
      
      row_shift_grid[r2,c2] = dy
      col_shift_grid[r2,c2] = dx
          
  row_shift_vect = np.max(row_shift_grid, axis=1)
  col_shift_vect = np.max(col_shift_grid, axis=0)
  for r in range(nr):
    print('row', r)
    rshift = 0 if r == 0 else np.sum(row_shift_vect[:r])
    for c in range(nc):
      print('\tcolumn', c)
      cshift = 0 if c == 0 else np.sum(col_shift_vect[:c])
      if r == 0 and c == 0: continue
      print('\t\tlocation', r, c)
      target_slide = sample_layout[r,c]
      if target_slide == 'pass':
        continue
      if target_slide is None:
        continue
      print('\t\ttarget slide:', target_slide)
      print(f'\t\tShifting by: rshift(1):{rshift}\tcshift(0):{cshift}')
      target_idx = labels==target_slide
      target_coords = coords[target_idx].copy()
      target_coords[:,0] -= min(target_coords[:,0])
      target_coords[:,1] -= min(target_coords[:,1])
      target_coords[:,0] += cshift
      target_coords[:,1] += rshift
      coords[target_idx] = target_coords
          
  # Flip dim1 (vertical dimension)
  coords[:,1] = -coords[:,1]
  return coords


def shift_coordinates(labels, coords, sample_layout):
  # do some gymnastics with the raw coordinates
  # we want to set the origin for each slide to (0,0)
  coords_out = coords.copy()
  u_labels = np.unique(labels)
  for l in u_labels:
    lidx = labels==l
    l_coords = coords_out[lidx,...]
    l_coords[:,0] = l_coords[:,0] - min(l_coords[:,0])
    l_coords[:,1] = l_coords[:,1] - min(l_coords[:,1])
    #l_coords[:,1] = -l_coords[:,1]
    coords_out[lidx,...] = l_coords
  
  nr,nc = sample_layout.shape
  print('creating coordinate layout:', nr,nc)
  
  for r2 in range(nr):
    if r2>0:
      row_shift = current_row_max # type: ignore
    else: 
      row_shift = 0
    
    current_row_max = 0
    for c2 in range(nc):
      if r2==0 and c2==0: continue
      
      target_slide = sample_layout[r2,c2]
      if target_slide == 'pass':
        continue
      if target_slide is None:
        continue
      
      ref_col = c2-1 if c2>0 else c2
      col_ref_slide = sample_layout[r2,ref_col] 
      
      target_idx = labels==target_slide
      target_coords = coords_out[target_idx].copy()
          
      target_coords[:,1] += row_shift
      if max(target_coords[:,1]) > current_row_max:
        current_row_max = max(target_coords[:,1])
      
      if col_ref_slide != target_slide:
        col_ref = coords_out[labels==col_ref_slide]
        col_max = max(col_ref[:,0])
        target_coords[:,0] += col_max
      
      coords_out[target_idx] = target_coords
          
  # Flip dim1 (vertical dimension)
  coords_out[:,1] = -coords_out[:,1]
  return coords_out


def hdf5_concat_v2(h5fs, h5out, channels=None, sample_layout=None, 
                   shuffle=False, seed=999):
  """ Concatenate several datasets
  We want to shuffle, without holding all the data in memory at once.
  """
  assert isinstance(h5fs, list)
  assert len(h5fs)>1

  if channels is None:
    channels = _get_hdf5_keys(h5fs[0], 'cells')
  else:
    assert isinstance(channels, list)

  fx = lambda x: os.path.splitext(os.path.basename(x))[0]
  samples = [fx(x) for x in h5fs]
  print('Merging: ', samples)

  with h5py.File(h5out, 'a') as h5fout:
    _make_source_dataset_v2(h5fout, h5fs, samples, f'cells/{channels[0]}', 'cell_samples',
                            shuffle=shuffle, seed=seed)
    _make_source_dataset_v2(h5fout, h5fs, samples, f'images/{channels[0]}', 'image_samples',
                            shuffle=shuffle, seed=seed)

    cell_size = _get_size_attr(h5fs[0], 'cells')
    for ch in channels:
      _merge_image_datasets_v2(h5fout, h5fs, 'cells', ch, cell_size, shuffle=shuffle, seed=seed)
      h5fout.flush()

    image_size = _get_size_attr(h5fs[0], 'images')
    for ch in channels:
      _merge_image_datasets_v2(h5fout, h5fs, 'images', ch, image_size, shuffle=shuffle, seed=seed)
      h5fout.flush()

    for ch in channels:
      _merge_value_datasets_v2(h5fout, h5fs, f'cell_membrane_stats/{ch}', stats=True, shuffle=shuffle, 
                               seed=seed, transfer_attr=['label'])
      _merge_value_datasets_v2(h5fout, h5fs, f'cell_nuclei_stats/{ch}', stats=True, shuffle=shuffle, 
                               seed=seed, transfer_attr=['label'])
      _merge_value_datasets_v2(h5fout, h5fs, f'tile_stats/{ch}', stats=True, shuffle=shuffle, 
                               seed=seed, transfer_attr=['label'])
      h5fout.flush()

    _merge_image_datasets_v2(h5fout, h5fs, 'meta', 'nuclear_masks', cell_size, shuffle=shuffle, seed=seed)
    _merge_image_datasets_v2(h5fout, h5fs, 'meta', 'membrane_masks', cell_size, shuffle=shuffle, seed=seed)
    _merge_value_datasets_v2(h5fout, h5fs, 'meta/Cell_IDs', stats=False, shuffle=shuffle, seed=seed)
    _merge_value_datasets_v2(h5fout, h5fs, 'meta/Tile_IDs', stats=False, shuffle=shuffle, seed=seed)
    h5fout.flush()

    _merge_coordinates_v2(h5fout, h5fs, 'meta/cell_coordinates', shuffle=shuffle, seed=seed)
    _merge_coordinates_v2(h5fout, h5fs, 'meta/tile_coordinates', shuffle=shuffle, seed=seed)

    _ = h5fout.create_dataset('meta/channel_names', data = np.array(channels, dtype='S'))

    # Apply coordinate shifting for convenient plotting later
    cell_labels = np.array([x.decode('utf-8') for x in h5fout['meta/cell_samples'][:]])
    cell_coords = h5fout['meta/cell_coordinates'][:]
    cell_shifted_coords = shift_coordinates_v2(cell_labels, cell_coords, sample_layout)
    _ = h5fout.create_dataset('meta/cell_coordinates_shift', data=cell_shifted_coords)

    image_labels = np.array([x.decode('utf-8') for x in h5fout['meta/image_samples'][:]])
    image_coords = h5fout['meta/tile_coordinates'][:]
    image_shifted_coords = shift_coordinates_v2(image_labels, image_coords, sample_layout)
    _ = h5fout.create_dataset('meta/tile_coordinates_shift', data=image_shifted_coords)

    h5fout.flush()




def hdf5_concat(h5fs, h5out, channels='all', sample_layout=None, dry_run=False):
  """ Concatenate several datasets

  Requires that the datasets all have the same number of channels
  and are of the same size (cell images, and tile images).

  Meta objects can be normally merged:
    - Cell_IDs
    - Tile_IDs
    - bounding_boxes
    - membrane_masks
    - nuclear_masks

  Meta objects that need to be treated specially:
    - cell_coordinates
    - tile_coordinates
  
  This takes a while, so we could do some kind of checkpointing by hashing
  hashes of the input files, and storing a success token along with each merged dataset.
  That way, one could check hashes of a set of input files with the token value
  and verify that the written file contains the expected data.

  The way with shuffling all data upfront is also very memory heavy. There can probably 
  be a refactor somewhere that lets us shuffle across all samples without having
  a whole stack of images in memory.
  

  Args:
    h5fs (list of str): Paths to hdf5 datasets to join
    h5out (str): Path where the joined hdf5 dataset will be created
    channels (str, list): if 'all', use all datasets, otherwise use 
      only sets from a provided list (TODO)
    sample_layout (np.ndarray, strings): 2D layout of samples after shifting coordinates
    dry_run (bool): if True, print info but do not execute file creation (TODO)
  
  Returns:
    Nothing
  """
  assert isinstance(h5fs, list)
  assert len(h5fs)>1

  if channels is None:
    channels = _get_hdf5_keys(h5fs[0], 'cells')
  else:
    assert isinstance(channels, list)

  fx = lambda x: os.path.splitext(os.path.basename(x))[0]
  samples = [fx(x) for x in h5fs]
  print('Merging: ', samples)

  print('Making source datasets')
  with h5py.File(h5out, 'a') as h5fout:
    _make_source_dataset(h5fout, h5fs, samples, f'cells/{channels[0]}', 'cell_samples')
    _make_source_dataset(h5fout, h5fs, samples, f'images/{channels[0]}', 'image_samples')

    cell_size = _get_size_attr(h5fs[0], 'cells')
    cell_perm = h5fout['meta/cell_samples_perm'][:]
    for ch in channels:
      _merge_image_datasets(h5fout, h5fs, 'cells', ch, cell_size, perm=cell_perm)
      h5fout.flush()

    image_size = _get_size_attr(h5fs[0], 'images')
    image_perm = h5fout['meta/image_samples_perm'][:]
    for ch in channels:
      _merge_image_datasets(h5fout, h5fs, 'images', ch, image_size, perm=image_perm)
      h5fout.flush()
 
    for ch in channels:
      _merge_value_datasets(h5fout, h5fs, f'cell_membrane_stats/{ch}', stats=True, perm=cell_perm, transfer_attr=['label'])
      _merge_value_datasets(h5fout, h5fs, f'cell_nuclei_stats/{ch}', stats=True, perm=cell_perm, transfer_attr=['label'])
      _merge_value_datasets(h5fout, h5fs, f'tile_stats/{ch}', stats=True, perm=image_perm, transfer_attr=['label'])
      h5fout.flush()

    # TODO what? these are missing? 
    # for ch in channels:
    #   _merge_value_datasets(h5fout, h5fs, f'tile_intensity/{ch}', stats=True, perm=image_perm)

    _merge_image_datasets(h5fout, h5fs, 'meta', 'nuclear_masks', cell_size, perm=cell_perm)
    _merge_image_datasets(h5fout, h5fs, 'meta', 'membrane_masks', cell_size, perm=cell_perm)
    _merge_value_datasets(h5fout, h5fs, 'meta/Cell_IDs', stats=False, perm=cell_perm)
    _merge_value_datasets(h5fout, h5fs, 'meta/Tile_IDs', stats=False, perm=image_perm)
    h5fout.flush()

    _merge_coordinates(h5fout, h5fs, 'meta/cell_coordinates', perm=cell_perm)
    _merge_coordinates(h5fout, h5fs, 'meta/tile_coordinates', perm=image_perm)

    _ = h5fout.create_dataset('meta/channel_names', data = np.array(channels, dtype='S'))

    # Apply coordinate shifting for convenient plotting later
    cell_labels = np.array([x.decode('utf-8') for x in h5fout['meta/cell_samples'][:]])
    cell_coords = h5fout['meta/cell_coordinates'][:]
    cell_shifted_coords = shift_coordinates_v2(cell_labels, cell_coords, sample_layout)
    _ = h5fout.create_dataset('meta/cell_coordinates_shift', data=cell_shifted_coords)

    image_labels = np.array([x.decode('utf-8') for x in h5fout['meta/image_samples'][:]])
    image_coords = h5fout['meta/tile_coordinates'][:]
    image_shifted_coords = shift_coordinates_v2(image_labels, image_coords, sample_layout)
    _ = h5fout.create_dataset('meta/tile_coordinates_shift', data=image_shifted_coords)

    h5fout.flush()