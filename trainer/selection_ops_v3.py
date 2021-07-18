import numpy as np
from matplotlib.colors import rgb2hex
import cv2

from micron2.codexutils import get_images, blend_images, load_nuclei_mask

from tifffile import imread
import zarr
import tqdm


def set_dropdown_menu(shared_variables, widgets):
  items = widgets['channels_select'].value
  widgets['focus_channel'].menu = items
  shared_variables['use_channels'] = items



def pull_images(image_sources, bbox, logger):
  logger.info(f'pulling images for bbox: {bbox}')
  images = {}
  for ch, src in image_sources.items():
    logger.info(f'pulling {ch} from image {src}')
    store = imread(src, aszarr=True)
    z = zarr.open(store, mode='r')
    img = z[bbox[2]:bbox[3], bbox[0]:bbox[1]]

    # in this case, flip up-down
    # img = img[::-1, :]
    images[ch] = img[::-1,:].copy()

  return images


# https://docs.bokeh.org/en/latest/docs/gallery/color_sliders.html
def hex_to_dec(hex):
  red = ''.join(hex.strip('#')[0:2])
  green = ''.join(hex.strip('#')[2:4])
  blue = ''.join(hex.strip('#')[4:6])
  return np.array((int(red, 16), int(green, 16), int(blue,16), 255))


def sample_nuclei(image_sources, coords, annotations, logger, sample_rate=0.005, wind=64):
  s = int(wind/2)
  val_inv = np.max(coords[:,0])
  # coords[:,0] = val_inv - coords[:,0]

  logger.info(f'Pulling nuclei from raw source images')

  # holds a (N x H x W) stack for each nucleus, on one channel.
  channel_stacks = {}

  has_annotation = annotations != ''
  has_annotation_inds = np.nonzero(has_annotation)[0]

  n_sample = int(coords.shape[0] * sample_rate)
  sample_inds = np.random.choice(coords.shape[0], size=n_sample, replace=False)
  sample_inds = list(set(sample_inds) - set(has_annotation_inds))

  sample_coords = coords[sample_inds]
  annotated_coords = coords[has_annotation]

  for ch, src in tqdm.tqdm(image_sources.items()):
    store = imread(src, aszarr=True)
    z = zarr.open(store, mode='r')
    channel_nuclei = []
    for r,c in sample_coords:
      bbox = [r-s, r+s, c-s, c+s]
      img = z[bbox[2]:bbox[3], bbox[0]:bbox[1]]
      channel_nuclei.append(img.copy())

    for r,c in annotated_coords:
      bbox = [r-s, r+s, c-s, c+s]
      img = z[bbox[2]:bbox[3], bbox[0]:bbox[1]]
      channel_nuclei.append(img.copy())

    channel_stacks[ch] = np.stack(channel_nuclei, axis=0)

  annot = ['']*len(sample_inds) + list(annotations[has_annotation])

  logger.info('Finished pulling nuclei:')
  for k,v in channel_stacks.items():
    logger.info(f'channel: {k} stack size: {v.shape} {v.dtype}')

  return channel_stacks, annot
  