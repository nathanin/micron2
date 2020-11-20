import numpy as np
import pytiff


def get_images(sources, bbox):
  """
  Get intensity images from the sources 

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



import seaborn as sns
def blend_images(images, saturation_vals=None, colors=None):
  """
  Color blend multiple intensity images into a single RGB image

  Args:
    images (np.uint8): (height, width, channels) the raw image data
    saturation_vals (list uint8): saturation values for each channel
    colors (list, np.ndarray): colors for each channel to gradate towards from black

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
    print(f'saturating image at value {sat_val}')
    img[img > sat_val] = sat_val
    img = img / sat_val
    
    view[:,:,0] += (img * colors[c,0]).astype(np.uint8)
    view[:,:,1] += (img * colors[c,1]).astype(np.uint8)
    view[:,:,2] += (img * colors[c,2]).astype(np.uint8)

  view[:,:,3] = 255 # Solid alpha?

  return blended
