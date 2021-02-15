import numpy as np
import pandas as pd
import h5py
import os

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

      dkeys = list(f[d].keys())
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


def _merge_image_datasets(h5fout, h5fs, dataset, channel, size):
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

  d = h5fout.create_dataset(f'{dataset}/{channel}', shape=(total_size,size,size), 
                            maxshape=(None,size,size),
                            chunks=(1,size,size), 
                            dtype='uint8', # TODO inherit dtype from the sources
                            compression='gzip')
  i=0
  for h5f in h5fs:
    with h5py.File(h5f,'r') as f:
      n=f[f'{dataset}/{channel}'].shape[0]
      d[i:i+n,...]=f[f'{dataset}/{channel}'][:]
    i=n


def _merge_value_datasets(h5fout, h5fs, dataset, stats):
  """ Merge scalar value datasets """
  values = []
  for h5f in h5fs:
    with h5py.File(h5f,'r') as f:
      values.append(f[dataset][:])
  values = np.concatenate(values)
  print(f'Merging scalar dataset: {dataset} new shape {values.shape}')
  d = h5fout.create_dataset(dataset, data=values)
  if stats:
    d.attrs['mean'] = np.mean(values)
    d.attrs['std'] = np.std(values)


def _make_source_dataset(h5fout, h5fs, samples, size_ref_dataset):
  """ Make a string dataset that lists the source sample for each slide """
  values = []
  for h5f, name in zip(h5fs, samples):
    with h5py.File(h5f,'r') as f:
      size = f[size_ref_dataset].shape[0]
    values += [name]*size
  d = h5fout.create_dataset('meta/samples', data=np.array(values, dtype='S'))


def hdf5_concat(h5fs, h5out, use_datasets='all', dry_run=False):
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

  Args:
    h5fs (list of str): Paths to hdf5 datasets to join
    h5out (str): Path where the joined hdf5 dataset will be created
    use_datasets (str, list): if 'all', use all datasets, otherwise use 
      only sets from a provided list (TODO)
    dry_run (bool): if True, print info but do not execute file creation (TODO)
  
  Returns:
    Nothing
  """
  assert isinstance(h5fs, list)
  assert len(h5fs)>1

  channels = _get_hdf5_keys(h5fs[0], 'cells')
  fx = lambda x: os.path.splitext(os.path.basename(x))[0]
  samples = [fx(x) for x in h5fs]
  print('Merging: ', samples)

  with h5py.File(h5out, 'w') as h5fout:
    _make_source_dataset(h5fout, h5fs, samples, f'cells/{channels[0]}')
    h5fout.flush()

    cell_size = _get_size_attr(h5fs[0], 'cells')
    for ch in channels:
      _merge_image_datasets(h5fout, h5fs, 'cells', ch, cell_size)
      h5fout.flush()

    image_size = _get_size_attr(h5fs[0], 'images')
    for ch in channels:
      _merge_image_datasets(h5fout, h5fs, 'images', ch, image_size)
      h5fout.flush()
 
    for ch in channels:
      _merge_value_datasets(h5fout, h5fs, f'cell_intensity/{ch}', stats=True)
      h5fout.flush()

    for ch in channels:
      _merge_value_datasets(h5fout, h5fs, f'tile_intensity/{ch}', stats=True)
      h5fout.flush()

    _merge_image_datasets(h5fout, h5fs, 'meta', 'nuclear_masks', cell_size)
    _merge_image_datasets(h5fout, h5fs, 'meta', 'membrane_masks', cell_size)
    _merge_value_datasets(h5fout, h5fs, 'meta/Cell_IDs', stats=False)
    _merge_value_datasets(h5fout, h5fs, 'meta/Tile_IDs', stats=False)

    # Need to merge coordinates as well. add a _merge_nd_datasets() function to handle NxM data
    h5fout.flush()

    d = h5fout.create_dataset('meta/channel_names', data = np.array(channels, dtype='S'))