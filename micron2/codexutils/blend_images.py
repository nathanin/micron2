import numpy as np

# import pytiff
from tifffile import imread as tif_imread
import zarr
import cv2

import seaborn as sns
import functools


def estimate_channel_background(source):
  """ Estimate the the low end of expression that should be suppressed """
  pass


def get_multiple_images(sources, bboxes):
  regions = {i: [] for i in range(len(bboxes))}
  for s in sources:
    store = tif_imread(s, aszarr=True)
    z = zarr.open(store, mode='r')
    for i,bbox in enumerate(bboxes):
      img_raw = z[bbox[0]:bbox[1], bbox[2]:bbox[3]]
      regions[i].append(img_raw)

    # with pytiff.Tiff(s, "r") as f:
    #   for i,bbox in enumerate(bboxes):
    #     img_raw = f.pages[0][bbox[0]:bbox[1], bbox[2]:bbox[3]]
    #     regions[i].append(img_raw)

  regions = [np.dstack(r) for i,r in regions.items()]
  return regions


# maxsize should be small; some images may be large 
@functools.lru_cache(maxsize=8)
def _load_image(s, a,b,c,d):

  store = tif_imread(s, aszarr=True)
  z = zarr.open(store, mode='r')
  x1=a
  x2=min(b, z.shape[0])
  y1=c
  y2=min(d, z.shape[1])
  img_raw = z[x1:x2, y1:y2]

  # with pytiff.Tiff(s, "r") as f:
  #   x1=a
  #   x2=min(b, f.pages[0].shape[0])
  #   y1=c
  #   y2=min(d, f.pages[0].shape[1])
  #   img_raw = f.pages[0][x1:x2, y1:y2]
  return img_raw


def get_images(sources, bbox, verbose=False):
  """ Get intensity images from the sources 

  Args:
    sources (list): files to read from
    bbox (list): bounding box as [x1, x2, y1, y2]

  Returns:
    images (np.uint8): (height, width, channels)
  """

  images = []
  for s in sources:
    if verbose:
      print(f'loading {bbox[0]} : {bbox[1]} and {bbox[2]} : {bbox[3]} from {s}')
    img_raw = _load_image(s, *bbox)

    if verbose:
      print(f'loaded image with mean: {np.mean(img_raw)} ({img_raw.dtype})')

    images.append(img_raw)

  return np.dstack(images)


def load_nuclei_mask(nuclei_path, bbox, dilation=1):
  """ Load nuclei mask and extract the borders

  Args:
    nuclei_path (str): where to find the image
    bbox (list): bounding box as [x1, x2, y1, y2]

  Returns:
    nuclei (np.bool): binary-like image that is true at nuclei borders
  """

  # with pytiff.Tiff(nuclei_path, "r") as f:
  #   nuclei = f.pages[0][bbox[0]:bbox[1], bbox[2]:bbox[3]]

  store = tif_imread(nuclei_path, aszarr=True)
  z = zarr.open(store, mode='r')
  img = z[bbox[0]:bbox[1], bbox[2]:bbox[3]]

  nuclei = (nuclei > 0).astype(np.uint8)
  kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
  dilation = cv2.dilate(nuclei,kern,iterations = dilation)
  nuclei = dilation - nuclei
  return nuclei > 0


def blend_images(images, saturation_vals=None, colors=None, 
                 nuclei=None, nuclei_color=None, verbose=False):
  """
  Color blend multiple intensity images into a single RGB image

  Args:
    images (np.uint8): (height, width, channels) the raw image data
    saturation_vals (list of tuples (uint8, uint8)): saturation values for each channel
    colors (list, np.ndarray): colors for each channel to gradate towards from black
    nuclei (np.ndarray, bool): a mask to apply after the image is blended
    nuclei_color (tuple, uint8): the color to use for nuclei mask

  Returns:
    blended (np.uint8): (height, width, 3) the RGB blended image
  """
  nc = images.shape[-1]
  if saturation_vals is None:
    # Default to 99%
    saturation_vals = []
    for c in range(nc):
      h = int(np.quantile(images[:,:,c], 0.99))
      l = int(h/256)
      saturation_vals.append((l, h))

  if verbose:
    for sv in saturation_vals:
      print(f'Saturation: {sv}')

  if colors is None:
    colors = (np.array(sns.color_palette(palette='tab20', n_colors=nc))*255).astype(np.uint8)

  h, w = images.shape[:2]
  blended = np.zeros((h,w), dtype=np.uint32)
  view = blended.view(dtype=np.uint8).reshape((h,w,4))
  # Loop 1. normalize all channels to [0,1]
  for c in range(nc):
    img = images[:,:,c]
    low_sat_val, high_sat_val = saturation_vals[c]
    high_sat_val = max(1, high_sat_val) # sometimes the 99th percentile can be 0
    img[img < low_sat_val] = 0
    img[img > high_sat_val] = high_sat_val
    img = img / high_sat_val
    img = (255 * img).astype(np.uint8) / 255.
    # images[:,:,c] = img
    
  # image_norm = np.max(images, axis=-1)

  # Loop 2. use the overall weights to properly color the image
  # for c in range(nc):
  #   img = images[:,:,c]
    # img = img/image_norm
    view[:,:,0] = np.clip(view[:,:,0] + (img * colors[c,0]), 0, 255).astype(np.uint8)
    view[:,:,1] = np.clip(view[:,:,1] + (img * colors[c,1]), 0, 255).astype(np.uint8)
    view[:,:,2] = np.clip(view[:,:,2] + (img * colors[c,2]), 0, 255).astype(np.uint8)

    # view[:,:,0] += (img * colors[c,0]).astype(np.uint8)
    # view[:,:,1] += (img * colors[c,1]).astype(np.uint8)
    # view[:,:,2] += (img * colors[c,2]).astype(np.uint8)

  view[:,:,3] = 255 # Solid alpha?


  if nuclei is not None:
    if nuclei.shape[0] != blended.shape[0]:
      nuclei = cv2.resize(nuclei.astype(np.uint8), (w,h), 
        interpolation=cv2.INTER_LINEAR)>0
    view[nuclei,0] = nuclei_color[0]
    view[nuclei,1] = nuclei_color[1]
    view[nuclei,2] = nuclei_color[2]

  return blended


