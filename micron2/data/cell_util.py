import numpy as np
import pandas as pd
import tqdm.auto as tqdm
import h5py
import warnings

try:
  import cv2
except:
  warnings.warn('Failed to import cv2')


""" Utilities for processing cell images/features """

kernel = np.ones((3,3))
# def staining_border_nonzero(h5f, channel, i, kernel=kernel):
  # x = h5f['cells'][channel][i,...]
  # m = h5f['meta']['nuclear_masks'][i,...]

def get_ring(m, kernel=kernel):
  md = cv2.dilate(m.astype(np.uint8),kernel,2) #dilate
  me = cv2.erode(m.astype(np.uint8),kernel,2) #also erode
  ring = md-me
  return ring


def staining_border_nonzero(x, m, kernel=kernel, threshold=20):
  """ Approximate ring coverage with the percent of non-zero border pixels """
  xthr = x>threshold
  if np.sum(xthr) == 0:
    return 0.
  md = cv2.dilate(m.astype(np.uint8),kernel,2) #dilate
  me = cv2.erode(m.astype(np.uint8),kernel,2) #also erode
  border_signal=xthr[(md-me)>0]
  return np.sum(border_signal)/len(border_signal)


def staining_rings(h5path, ring_channels, obs_names):
  """ Apply staining border function """

  ring_positive_pct = pd.DataFrame(index=obs_names,
                                    columns=[f'{ch}_ringpct' for ch in ring_channels],
                                    dtype=np.float32)

  with h5py.File(h5path, 'r') as h5f:
    ncells = len(obs_names)
    h5_ncells = h5f['cell_intensity']['DAPI'].shape[0]
    assert ncells==h5_ncells

    for i in tqdm.trange(h5_ncells):
      m = h5f['meta']['nuclear_masks'][i,...]
      vect = []
      for ch in ring_channels:
        x = h5f['cells'][ch][i,...]
        v = staining_border_nonzero(x,m)
        vect.append(v)
      ring_positive_pct.loc[obs_names[i],:] = vect

  return ring_positive_pct
