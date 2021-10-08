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



def pull_images(image_sources, bbox, logger, resize=1.):
  logger.info(f'pulling images for bbox: {bbox}')
  images = {}
  for ch, src in image_sources.items():
    logger.info(f'pulling {ch} from image {src}')
    store = imread(src, aszarr=True)
    z = zarr.open(store, mode='r')

    if (bbox is None) or (bbox[3] >= z.shape[0]) or (bbox[1] >= z.shape[1]):
      img = z[:]
    
    else:
      img = z[bbox[2]:bbox[3], bbox[0]:bbox[1]]

    # in this case, flip up-down
    # img = img[::-1, :]
    if resize!=1.:
      img = cv2.resize(img, dsize=(0,0), fx=resize, fy=resize,
                       interpolation=cv2.INTER_LINEAR)

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

  target_size = (wind, wind)

  for ch, src in tqdm.tqdm(image_sources.items()):
    store = imread(src, aszarr=True)
    z = zarr.open(store, mode='r')
    max_row = z.shape[0]
    max_col = z.shape[1]
    channel_nuclei = []
    for r,c in sample_coords:
      bbox = [r-s, r+s, c-s, c+s]
      # if bbox[0] <= 0: continue
      # if bbox[1] >= max_row: continue
      # if bbox[2] <= 0: continue
      # if bbox[3] >= max_col: continue
      img = z[bbox[2]:bbox[3], bbox[0]:bbox[1]]
      if img.shape != target_size:
        continue
      channel_nuclei.append(img.copy())
    valid_nuclei = len(channel_nuclei)

    for r,c in annotated_coords:
      bbox = [r-s, r+s, c-s, c+s]
      # if bbox[0] <= 0: continue
      # if bbox[1] >= max_row: continue
      # if bbox[2] <= 0: continue
      # if bbox[3] >= max_col: continue
      img = z[bbox[2]:bbox[3], bbox[0]:bbox[1]]
      # if img.shape != target_size:
      #   continue
      channel_nuclei.append(img.copy())

    channel_stacks[ch] = np.stack(channel_nuclei, axis=0)

  annot = ['']*valid_nuclei + list(annotations[has_annotation])

  logger.info('Finished pulling nuclei:')
  for k,v in channel_stacks.items():
    logger.info(f'channel: {k} stack size: {v.shape} {v.dtype}')

  return channel_stacks, annot
  

def gather_features(image_sources, feature_df, ):
  pass