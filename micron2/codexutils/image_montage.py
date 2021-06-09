import cv2
import numpy as np
from numpy.lib.npyio import load
from .blend_images import get_images, get_multiple_images, blend_images, load_nuclei_mask


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


def layout_cells(montage, layer, ncol=3, padding=1, padval=250, snake=True):
  """
  snaked:
  1 2 3 4
  8 7 6 5
  9 

  non-snaked:
  1 2 3 4
  5 6 7 8 
  9
  """
  n_regions = len(montage[0])
  nrow = int(np.ceil(n_regions/ncol))
  
  rows = []
  i = 0
  expected_shape = montage[layer][0].shape
  for r in range(nrow):
    row = []
    for c in range(ncol):
      if i >= n_regions:
        if snake & (r%2!=0):
          row.insert(0, np.zeros_like(montage[0][0]))
        else:
          row.append(np.zeros_like(montage[0][0]))
      else:
        img = montage[layer][i].copy()
        img = outline(img, padding, padval)
        
        if img.shape[0] < expected_shape[0]:
          d = expected_shape[0] - img.shape[0] 
          p = np.zeros((d, img.shape[1], 3), dtype=img.dtype)
          img = np.concatenate([img, p], axis=0)

        if img.shape[1] < expected_shape[1]:
          d = expected_shape[1] - img.shape[1] 
          p = np.zeros((img.shape[0], d, 3), dtype=img.dtype)
          img = np.concatenate([img, p], axis=1)

        # print(f'Row {r} Adding image column {c}: {img.shape}')

        if snake & (r%2!=0):
          row.insert(0, img)
        else:
          row.append(img)
      i+=1

    row = np.hstack(row)
    # print(f'built image row {r}: {row.shape}')
    rows.append(row)
  out = np.vstack(rows)
  return out


class ImageMontage:
  def __init__(self, sources={}, colors={}, saturations={}, channel_groups=[], dilation=1, verbose=False):
    """
    Args:
      sources (dict): dictionary mapping channel name to image
      colors (dict): dictionary mapping channel name to color
      saturations (dict): dictionary mapping channel name to tuple (low, high) saturation values
    """
    self.sources = sources
    self.colors = colors
    self.saturations = saturations
    self.channel_groups = channel_groups
    self.verbose = verbose
    self.dilation = dilation


  def overlay_solid(self, view, bbox, annotation, downsample=None):
    color = hex_to_dec(self.colors[annotation])
    a = load_nuclei_mask(self.sources[annotation], bbox, dilation=self.dilation)

    r,g,b = np.split(view, 3, axis=-1)
    r[a] = color[0]
    g[a] = color[1]
    b[a] = color[2]
    view = np.dstack([r,g,b])

    return view


  def montage(self, bbox, downsample=None):
    m = []
    for group in self.channel_groups:
      if 'nuclei' in group:
        group = [g for g in group if g != 'nuclei']
        do_nuclei = True
      else:
        do_nuclei = False
      
      if 'membrane' in group:
        group = [g for g in group if g != 'membrane']
        do_membrane = True
      else:
        do_membrane = False

      if self.verbose:
        print(f'Making group: {group}')

      group_sources = [self.sources[ch] for ch in group]
      group_colors = np.stack([hex_to_dec(self.colors[ch]) for ch in group], axis=0)
      group_saturations = [self.saturations[ch] for ch in group]
      images = get_images(group_sources, bbox, verbose=self.verbose)
      if downsample is not None:
        images = cv2.resize(images, dsize=(0,0), fx=downsample, fy=downsample)
        if len(group) == 1:
          images = np.expand_dims(images, axis=-1)
      if self.verbose:
        print(f'Image stack: {images.shape}')
      blended = blend_images(images, saturation_vals=group_saturations, colors=group_colors)
      h,w = images.shape[:2]
      view = blended.view(dtype=np.uint8).reshape((h,w,4)).copy()[:,:,:3]

      if do_nuclei:
        view = self.overlay_solid(view, bbox, 'nuclei', downsample=downsample)

      if do_membrane:
        view = self.overlay_solid(view, bbox, 'membrane', downsample=downsample)

      m.append(view)
    return m


  def montage_several(self, bboxes):
    """
    Pass in a list of bboxes corresponding to, e.g. a set of cells
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


  def add_color_legend(self, image, group_num, fontScale=1, legend_ht=64, label_space=24, 
                       thickness=2, loc='legend', xstart=36, font=cv2.FONT_HERSHEY_PLAIN,
                       print_saturation=False
                       ):
    group = self.channel_groups[group_num] 
    xval = xstart
    # Add legend to the bottom of the image (outside the image area)
    if loc=='legend':
      legend = np.zeros((legend_ht, image.shape[1], 3), dtype=np.uint8)
      ht = int(legend_ht * 0.8)
      for i, ch in enumerate(group):
        color = hex_to_dec(self.colors[ch])
        cv2.putText(legend, ch, (xval, ht), font, fontScale, color=color, 
                    thickness=thickness)
        xval += label_space
      image_out = np.vstack([image, legend])

    elif loc=='corner':
      image_out = image.copy()
      ht = int(image.shape[0]-xstart)
      for i, ch in enumerate(group):
        color = hex_to_dec(self.colors[ch])
        sat = f' ({self.saturations[ch][1]})' if print_saturation else ''
        txt = f'{ch}' + sat
        cv2.putText(image_out, txt, (xval, ht), font, fontScale, color=color, 
                    thickness=thickness)
        xval += label_space

    return image_out

