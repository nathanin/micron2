import numpy as np
import pandas as pd
import time
import h5py
import os
import warnings

try:
  import pytiff
except:
  warnings.warn('Failed to import pytiff.')

import tensorflow as tf
try:
  import tensorflow_io as tfio
except:
  warnings.warn('Failed to load tensorflow-io.')

from tqdm.auto import tqdm

"""
We want to load subsets of data from the hdf5 file structure

There can be a few possibilities:
1. Load all instances into memory (`load_dataset` function).
2. Construct a view into the hdf5 that lets us stream data.

Situation #2 is important to build because we're building out 
datasets that, at full size, will be large and hard to work with
in-memory.

1 experiment's worth of 150K cells and 40 channels is 1.1GB on disk (uint8, with gzip compression).

The goal is to have a loading scheme that will allow processing >1M cells and 40 channels
on a machine with <32GB RAM. 

"""


def load_dataset(fpath, use_channels=['DAPI', 'CD45', 'PanCytoK'], verbose=False):
  """
  Load channels from dataset
  """

  f = h5py.File('tests/dataset.hdf5', 'r')
  channels = [j.decode('UTF-8') for j in f['meta/channel_names'][:]]

  x = []
  for c in use_channels:
    t1 = time.time()
    x.append(f[f'cells/{c}'][:])
    t2 = time.time()
    dt = t2-t1
    if verbose:
      print(f'loaded {c:<10}\t{dt:3.3f}')
  x = np.stack(x, axis=-1)
  f.close()
  return x


def process_channels(*x):
  # x = [tf.cast(x_ / 255, tf.uint8) for x_ in x]
  x = tf.stack(x, axis=-1)
  #x = x  / tf.reduce_max(x)
  return x


def stream_dataset(fpath, use_channels=['DAPI', 'CD45', 'PanCytoK'], group_name='cells'):
  """
  Set up streaming from an hdf5 dataset

  Returns a tensorflow.data.Dataset object
  Layer on shuffle and perturb transformations later

  We need to extend this to ingest multiple source files at a time.

  Args:
    fpath (str): file name (put it on a fast device)
    use_channels (list): datasets found under the group name
    group_name (str): top level group to use

  Returns:
    dataset (tf.data.Dataset)

  """
  print(f'setting up dataset from {fpath} with channels {use_channels}')
  spec = tf.uint8
  channel_ds = [tfio.IODataset.from_hdf5(fpath, f'/{group_name}/{c}', spec=spec) for c in use_channels]
  dataset = (tf.data.Dataset.zip(tuple(channel_ds))
            .map(process_channels)
            )

  return dataset
  

AUTO = tf.data.experimental.AUTOTUNE
def stream_dataset_parallel(fpaths, use_channels=['DAPI', 'CD45', 'PanCytoK'], group_name='cells'):
  dataset = tf.data.Dataset.from_tensor_slices(fpaths)
  def load_fn(fpath):
    return stream_dataset(fpath, use_channels, group_name=group_name).repeat()

  dataset = dataset.interleave(lambda x: load_fn(x), 
                               cycle_length=len(fpaths), 
                               block_length=4, num_parallel_calls=AUTO)
  return dataset


def stream_dataset_images(image_stack):
  print(f'setting up streaming from images: {image_stack.shape}')
  dataset = tf.data.Dataset.from_tensor_slices(image_stack)

  return dataset