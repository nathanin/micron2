import numpy as np
import pandas as pd
import pytiff
import time
import h5py
import os
import tensorflow as tf
import tensorflow_io as tfio
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

The goal is to have a loading scheme that will allow >1M cells and 40 channels
on a machine with 32GB RAM. 

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
  #x = [x_ / tf.reduce_max(x_) for x_ in x]
  x = tf.stack(x, axis=-1)
  #x = x  / tf.reduce_max(x)
  return x

def stream_dataset(fpath, use_channels=['DAPI', 'CD45', 'PanCytoK']):
  """
  Set up streaming from an hdf5 dataset

  Returns a tensorflow.data.Dataset object
  Layer on shuffle and perturb transformations later

  We need to extend this to ingest multiple source files at a time.
  """
  channel_ds = [tfio.IODataset.from_hdf5(fpath, f'/cells/{c}') for c in use_channels]
  dataset = (tf.data.Dataset.zip(tuple(channel_ds))
            .map(process_channels))

  return dataset
  