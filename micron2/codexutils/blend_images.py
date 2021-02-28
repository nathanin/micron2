import numpy as np

import pytiff
import cv2

import seaborn as sns


def estimate_channel_background(source):
  """ Estimate the the low end of expression that should be suppressed """
  pass


def get_images(sources, bbox):
  """ Get intensity images from the sources 

  Args:
    sources (list): files to read from
    bbox (list): bounding box as [x1, x2, y1, y2]

  Returns:
    images (np.uint8): (height, width, channels)
  """

  images = []
  for s in sources:
    with pytiff.Tiff(s, "r") as f:
      print(f'loading {bbox[0]} : {bbox[1]} and {bbox[2]} : {bbox[3]}')
      img_raw = f.pages[0][bbox[0]:bbox[1], bbox[2]:bbox[3]]

    # ## Auto threshold based on image histogram
    # N, bins = np.histogram(img_raw.ravel(), 256)
    # npix = np.cumsum(N)
    # target = np.prod(img_raw.shape) / 10
    # bin_index = np.argwhere(npix > target)[0,0]
    # low_cutoff = int(bins[bin_index+1])
    # low_cutoff = max(low_cutoff, 15)

    # # ## Auto threshold to max / 256
    # # low_cutoff = np.max(img_raw) / 256
    # print(f'Applying cutoff of {low_cutoff} to {s.split("/")[-1]}')

    # img_raw[img_raw < low_cutoff] = 0
    # img = np.ceil(255 * (img_raw / 2**16)).astype(np.uint8)
    # img = (255 * (img_raw / 2**16)).astype(np.uint8)

    images.append(img_raw)

  return np.dstack(images)


def load_nuclei_mask(nuclei_path, bbox):
  """ Load nuclei mask and extract the borders

  Args:
    nuclei_path (str): where to find the image
    bbox (list): bounding box as [x1, x2, y1, y2]

  Returns:
    nuclei (np.bool): binary-like image that is true at nuclei borders
  """

  with pytiff.Tiff(nuclei_path, "r") as f:
    nuclei = f.pages[0][bbox[0]:bbox[1], bbox[2]:bbox[3]]

  nuclei = (nuclei > 0).astype(np.uint8)
  kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
  dilation = cv2.dilate(nuclei,kern,iterations = 1)
  nuclei = dilation - nuclei
  return nuclei > 0


def blend_images(images, saturation_vals=None, colors=None, 
                 nuclei=None, nuclei_color=None):
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

  if colors is None:
    colors = (np.array(sns.color_palette(palette='tab20', n_colors=nc))*255).astype(np.uint8)
    # print()

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


