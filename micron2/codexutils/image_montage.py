import cv2
import numpy as np
from .blend_images import get_images, get_multiple_images, blend_images

# https://docs.bokeh.org/en/latest/docs/gallery/color_sliders.html
def hex_to_dec(hex):
  red = ''.join(hex.strip('#')[0:2])
  green = ''.join(hex.strip('#')[2:4])
  blue = ''.join(hex.strip('#')[4:6])
  return (int(red, 16), int(green, 16), int(blue,16), 255)

def outline(img, padding, padval):
  img[:padding,:,:] = padval
  img[-padding:,:,:] = padval
  img[:,:padding,:] = padval
  img[:,-padding:,:] = padval
  return img

def layout_cells(montage, layer, ncol=3, padding=1, padval=250):
  n_regions = len(montage[0])
  nrow = int(np.ceil(n_regions/ncol))
  
  rows = []
  i = 0
  for r in range(nrow):
    row = []
    for c in range(ncol):
      if i >= n_regions:
        row.append(np.zeros_like(montage[0][0]))
      else:
        img = montage[layer][i].copy()
        img = outline(img, padding, padval)
        row.append(img)
      i+=1
    row = np.hstack(row)
    rows.append(row)
  out = np.vstack(rows)
  return out


class ImageMontage:
  def __init__(self, sources={}, colors={}, saturations={}, channel_groups=[]):
    """
    Args:
      sources (dict): dictionary mapping channel name to image
      colors (dict): dictionary mappig channel name to color
      saturations (dict): dictionary mapping channel name to tuple (low, high) saturation values
    """
    self.sources = sources
    self.colors = colors
    self.saturations = saturations
    self.channel_groups = channel_groups


  def montage(self, bbox):
    m = []
    for group in self.channel_groups:
      group_sources = [self.sources[ch] for ch in group]
      group_colors = np.stack([hex_to_dec(self.colors[ch]) for ch in group], axis=0)
      group_saturations = [self.saturations[ch] for ch in group]

      images = get_images(group_sources, bbox)
      blended = blend_images(images, saturation_vals=group_saturations, colors=group_colors)

      h,w = images.shape[:2]
      view = blended.view(dtype=np.uint8).reshape((h,w,4)).copy()[:,:,:3]

      m.append(view)

    return m


  def montage_several(self, bboxes):
    """
    m ~ [region1, region2, ...]
    region1 ~ [group1, group2, ...]
    """
    m = []
    for group in self.channel_groups:
      group_sources = [self.sources[ch] for ch in group]
      group_colors = np.stack([hex_to_dec(self.colors[ch]) for ch in group], axis=0)
      group_saturations = [self.saturations[ch] for ch in group]

      regions = get_multiple_images(group_sources, bboxes)

      blended_regions = []
      for region in regions:
        blended = blend_images(region, saturation_vals=group_saturations, colors=group_colors)
        h,w = blended.shape[:2]
        view = blended.view(dtype=np.uint8).reshape((h,w,4)).copy()[:,:,:3]
        blended_regions.append(view)

      m.append(blended_regions)

    return m


  def add_color_legend(self, image, group_num, fontScale=1):
    legend = np.zeros((32, image.shape[1], 3), dtype=np.uint8)
    group = self.channel_groups[group_num] 
    xval = 24
    font = cv2.FONT_HERSHEY_PLAIN
    for i, ch in enumerate(group):
      color = hex_to_dec(self.colors[ch])
      print(ch, color)
      cv2.putText(legend, ch, (xval, 24), font, fontScale, color=color, thickness=2)
      xval += 128

    image_out = np.vstack([image, legend])
    return image_out

