import numpy as np

import pytiff
import cv2


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
      img = f.pages[0][bbox[0]:bbox[1], bbox[2]:bbox[3]]
      img = (255 * (img / 2**16)).astype(np.uint8)
    images.append(img)

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


import seaborn as sns
def blend_images(images, saturation_vals=None, colors=None, 
                 nuclei=None, nuclei_color=None):
  """
  Color blend multiple intensity images into a single RGB image

  Args:
    images (np.uint8): (height, width, channels) the raw image data
    saturation_vals (list uint8): saturation values for each channel
    colors (list, np.ndarray): colors for each channel to gradate towards from black
    nuclei (np.ndarray, bool): a mask to apply after the image is blended
    nuclei_color (tuple, uint8): the color to use for nuclei mask

  Returns:
    blended (np.uint8): (height, width, 3) the RGB blended image
  """
  nc = images.shape[-1]
  if saturation_vals is None:
    # Default to 95%
    saturation_vals = [np.quantile(images[:,:,c], 0.99).astype(np.uint8) for c in range(nc)]

  if colors is None:
    colors = (np.array(sns.color_palette(palette='tab20', n_colors=nc))*255).astype(np.uint8)
    # print()

  h, w = images.shape[:2]
  blended = np.zeros((h,w), dtype=np.uint32)
  view = blended.view(dtype=np.uint8).reshape((h,w,4))
  for c in range(nc):
    img = images[:,:,c]
    sat_val = saturation_vals[c]
    img[img > sat_val] = sat_val
    img = img / sat_val
    
    view[:,:,0] += (img * colors[c,0]).astype(np.uint8)
    view[:,:,1] += (img * colors[c,1]).astype(np.uint8)
    view[:,:,2] += (img * colors[c,2]).astype(np.uint8)

  view[:,:,3] = 255 # Solid alpha?

  if nuclei is not None:
    view[nuclei,0] = nuclei_color[0]
    view[nuclei,1] = nuclei_color[1]
    view[nuclei,2] = nuclei_color[2]

  return blended


