import numpy as np
from bokeh.models import (ColumnDataSource, BoxSelectTool, LassoSelectTool, Button, 
                          Div, Select, Slider, TextInput, MultiChoice, ColorPicker, )
import logging
from matplotlib.colors import rgb2hex
from bokeh.models import Range1d

from micron2.codexutils import get_images, blend_images

# def make_logger():
logger = logging.getLogger('viewer')
logger.setLevel('INFO')
print(logger.handlers)


def set_dropdown_menu(shared_variables, widgets):
  items = widgets['channels_select'].value
  widgets['focus_channel'].menu = items
  shared_variables['use_channels'] = items


def update_edit_hist(shared_variables, widgets, figure_sources, figures):
  _ = maybe_pull(shared_variables['use_channels'],
                 shared_variables['active_raw_images'],
                 shared_variables['image_sources'],
                 shared_variables['bbox'])

  ac = shared_variables['active_channel'] 
  logger.info(f'Updating intenssity histogram for channel {ac}')
  if shared_variables['active_raw_images'][ac] is None:
    logger.info(f'Empty active channel {active_channel[0]}')
    return

  img = shared_variables['active_raw_images'][ac].ravel()
  hist, edges = np.histogram(img, bins=shared_variables['nbins'])
  figure_sources['intensity_hist'].data = {'value': np.log1p(hist), 'x': edges[:-1]}
  figures['intensity_hist'].title.text = f'{ac} intensity'


def set_active_channel(event, shared_variables, widgets, figure_sources, figures):
  ac = event.item
  shared_variables['active_channel'] = ac
  logger.info(f'setting active channel: {ac}')

  widgets['focus_channel'].label = f'Edit {ac} image'
  widgets['color_picker'].title = f'{ac} color'
  widgets['color_picker'].color = rgb2hex(shared_variables['channel_colors'][ac])
  widgets['color_slider'].title = f'{ac} saturation'
  widgets['color_slider'].value_throttled = shared_variables['saturation_vals'][ac]
  widgets['color_slider'].value = shared_variables['saturation_vals'][ac]

  shared_variables['use_channels'] = widgets['channels_select'].value
  update_edit_hist(shared_variables, widgets, figure_sources, figures)


def maybe_pull(use_channels, active_raw_images, image_sources, bbox):
  ## Decide if we have to read images from disk or not
  need_to_pull = False
  for c in use_channels:
    img = active_raw_images[c]
    if img is None:
      logger.info(f'channel {c} needs to be pulled')
      need_to_pull = True
      break
        
  if need_to_pull:
    logger.info(f'Pulling bbox: {bbox} from {use_channels}')
    use_files = [image_sources[c] for c in use_channels]
    images = get_images(use_files, bbox)
  else:
    images = np.dstack([active_raw_images[c] for c in use_channels])

  for i, c in enumerate(use_channels):
    active_raw_images[c] = images[:,:,i].copy()
  return images


def update_image_plot(shared_variables, widgets, figure_sources, figures):
  use_channels = shared_variables['use_channels']
  channel_colors = shared_variables['channel_colors']
  saturation_vals = shared_variables['saturation_vals']
  bbox = shared_variables['bbox']


  if len(use_channels) == 0:
    logger.info('No channels selected')
    return

  if sum(bbox) == 0:
    logger.info('No bbox to draw')
    return

  images = maybe_pull(shared_variables['use_channels'],
                      shared_variables['active_raw_images'],
                      shared_variables['image_sources'],
                      shared_variables['bbox'])

  colors = []
  for c in use_channels:
    chc = (np.array(channel_colors[c])*255).astype(np.uint8)
    logger.info(f'setting color {c} {chc}')
    colors.append(chc)

  colors = np.stack(colors, axis=0)
  saturation = []
  for c in use_channels:
    s = saturation_vals[c] 
    if s is None:
      s = np.quantile(s, 0.95)
    saturation.append(s)

  blended = blend_images(images, saturation_vals=saturation, colors=colors)
  blended = blended[::-1,:] # flip

  ## Set the aspect ratio according to the selected area
  dw = bbox[1] - bbox[0]
  dh = bbox[3] - bbox[2]
  y = dw / dh
  logger.info(f'setting image height: {y}')
  figures['image_plot'].y_range.end = y#Range1d(0, y)
  # figures['image_plot'].dh = y

  figure_sources['image_data'].data = {'value': [blended],
                                       'dw': [1/y],
                                       'dh': [1],}


def update_bbox(inds, shared_variables):
  bbox = shared_variables['bbox']
  use_channels = shared_variables['use_channels']
  active_raw_images = shared_variables['active_raw_images']
  coords = shared_variables['coords']

  xmin = min(coords[inds,1])
  xmax = max(coords[inds,1])
  ymin = min(coords[inds,0])
  ymax = max(coords[inds,0])

  # bbox = [xmin, xmax, ymin, ymax]
  bbox[0] = xmin
  bbox[1] = xmax
  bbox[2] = ymin
  bbox[3] = ymax

  # reset active images
  for c in use_channels:
    active_raw_images[c] = None