import numpy as np
from bokeh.models import (ColumnDataSource, BoxSelectTool, LassoSelectTool, Button, 
                          Div, Select, Slider, TextInput, MultiChoice, ColorPicker, )
import logging
from matplotlib.colors import rgb2hex
from bokeh.models import Range1d

from micron2.codexutils import get_images, blend_images, load_nuclei_mask

# def make_logger():
logger = logging.getLogger('viewer')
logger.setLevel('INFO')
print(logger.handlers)


def set_dropdown_menu(shared_variables, widgets):
  items = widgets['channels_select'].value
  widgets['focus_channel'].menu = items
  shared_variables['use_channels'] = items


# def update_edit_hist(shared_variables, widgets, figure_sources, figures):
#   _ = maybe_pull(shared_variables['use_channels'],
#                  shared_variables['active_raw_images'],
#                  shared_variables['image_sources'],
#                  shared_variables['bbox'])

#   ac = shared_variables['active_channel'] 
#   logger.info(f'Updating intenssity histogram for channel {ac}')
#   if shared_variables['active_raw_images'][ac] is None:
#     logger.info(f'Empty active channel {active_channel[0]}')
#     return

#   img = shared_variables['active_raw_images'][ac].ravel()
#   hist, edges = np.histogram(img, bins=shared_variables['nbins'])
#   figure_sources['intensity_hist'].data = {'value': np.log1p(hist), 'x': edges[:-1]}
#   figures['intensity_hist'].title.text = f'{ac} intensity'


# def set_active_channel(event, shared_variables, widgets, figure_sources, figures):
#   ac = event.item
#   shared_variables['active_channel'] = ac
#   logger.info(f'setting active channel: {ac}')

#   widgets['focus_channel'].label = f'Edit {ac} image'
#   widgets['color_picker'].title = f'{ac} color'
#   widgets['color_picker'].color = rgb2hex(shared_variables['channel_colors'][ac])
#   widgets['color_saturation'].title = f'{ac} saturation'
#   widgets['color_saturation'].value = shared_variables['saturation_vals'][ac]

#   shared_variables['use_channels'] = widgets['channels_select'].value
#   update_edit_hist(shared_variables, widgets, figure_sources, figures)


def maybe_pull(use_channels, active_raw_images, image_sources, bbox):
  ## Decide if we have to read images from disk or not
  need_to_pull = False
  for c in use_channels:
    img = active_raw_images[c]
    if img is None:
      logger.info(f'channel {c} needs to be pulled')
      need_to_pull = True
      break
        
  use_files = [image_sources[c] for c in use_channels]
  if need_to_pull:
    logger.info(f'Pulling bbox: {bbox} from {use_channels}')
    images = get_images(use_files, bbox)
  else:
    try:
      images = np.dstack([active_raw_images[c] for c in use_channels])
    except:
      images = get_images(use_files, bbox)

  for i, c in enumerate(use_channels):
    active_raw_images[c] = images[:,:,i].copy()
  return images


def _get_active_image_channels(widgets):
  use_channels = []
  for k, w in widgets.items():
    if 'nuclei' in k:
      continue
    if 'focus_channel' in k:
      ch = w.label
      if w.active:
        logger.info(f'requested to draw channel {ch}')
        use_channels.append(ch)
  return use_channels

# https://docs.bokeh.org/en/latest/docs/gallery/color_sliders.html
def hex_to_dec(hex):
  red = ''.join(hex.strip('#')[0:2])
  green = ''.join(hex.strip('#')[2:4])
  blue = ''.join(hex.strip('#')[4:6])
  return np.array((int(red, 16), int(green, 16), int(blue,16), 255))


def update_image_plot(shared_variables, widgets, figure_sources, figures):
  # use_channels = shared_variables['use_channels']
  use_channels = _get_active_image_channels(widgets)
  channel_colors = shared_variables['channel_colors']
  saturation_vals = shared_variables['saturation_vals']
  bbox = shared_variables['bbox']
  bbox_plot = shared_variables['bbox_plot']


  if len(use_channels) == 0:
    logger.info('No channels selected')
    return

  if sum(bbox) == 0:
    logger.info('No bbox to draw')
    return

  images = maybe_pull(use_channels,
                      shared_variables['active_raw_images'],
                      shared_variables['image_sources'],
                      shared_variables['bbox'])

  colors = []
  for c in use_channels:
    chc = (np.array(channel_colors[c])*255).astype(np.uint8)
    logger.info(f'drawing channel {c} with color {chc}')
    colors.append(chc)

  colors = np.stack(colors, axis=0)
  saturation = []
  for i,c in enumerate(use_channels):
    slow, shigh = saturation_vals[c] 

    if (slow is None) or (slow == 0):
      img = images[:,:,i]
      if img.sum() == 0:
        slow = 0
      else:
        slow = int(np.max(img)/256)

    if (shigh is None) or (shigh == 0):
      img = images[:,:,i]
      if img.sum() == 0:
        shigh = 0
      else:
        vals = img.ravel()[img.ravel()>0]
        shigh = np.quantile(vals, 0.99)

    #slow = int(shigh / 256)

    # make sure the saturation val widget reflects the value being drawn
    widgets[f'color_saturation_{c}_low'].value = slow
    widgets[f'color_saturation_{c}_high'].value = shigh

    saturation.append((slow,shigh))

  if widgets['focus_channel_nuclei'].active:
    nuclei = load_nuclei_mask(shared_variables['nuclei_path'], shared_variables['bbox'])
    nuclei_color = widgets['color_picker_nuclei'].color
    nuclei_color = hex_to_dec(nuclei_color)
    logger.info(f'drawing nuclei with color {nuclei_color}')
  else:
    nuclei = None
    nuclei_color = None
  blended = blend_images(images, saturation_vals=saturation, colors=colors, 
                         nuclei=nuclei, nuclei_color=nuclei_color)
  blended = blended[::-1,:] # flip
  logger.info(f'blended image: {blended.shape}')

  ## Set the aspect ratio according to the selected area
  dw = bbox[3] - bbox[2]
  dh = bbox[1] - bbox[0]
  logger.info(f'bbox would suggest the image shape to be: {dw} x {dh}')

  # y = dw / dh
  # logger.info(f'setting image height: {y}')
  # figures['image_plot'].y_range.end = y#Range1d(0, y)
  # figures['image_plot'].dh = y

  # figures['image_plot'].x_range = figures['scatter_plot'].x_range
  # figures['image_plot'].y_range = figures['scatter_plot'].y_range

  logger.info(f'update image aspect ratio: {dw} / {dh}')
  figure_sources['image_data'].data = {'value': [blended],
                                      ## Aspect ratio preserved, not matching scatter coordinates
                                      #  'dw': [1/y],
                                      #  'dh': [1],
                                      ## Aspect ratio preserved, matching scatter coordinates
                                      ## This would be preferred if the scatter plot would guarantee
                                      ##  preservation of the aspect ratio
                                      'dw': [dw], 
                                      'dh': [dh],
                                      ## Fill the square
                                      # 'dw': [max(dw,dh)], 
                                      # 'dh': [max(dw,dh)],
                                      'x0': [bbox_plot[0]],
                                      'y0': [bbox_plot[2]],
                                      # 'x0': [0],
                                      # 'y0': [0],
                                       }

  logger.info('Done pushing data for updated image.')



def update_bbox(shared_variables, figures):
  bbox = shared_variables['bbox']
  logger.info(f'BBOX updating bbox: old bbox: {bbox}')
  use_channels = shared_variables['use_channels']
  active_raw_images = shared_variables['active_raw_images']
  coords = shared_variables['coords']

  box_selection = shared_variables['box_selection']

  # These are the 'real' coordinates, to be used for pulling from the image
  xmin = min(coords[box_selection,1])
  xmax = max(coords[box_selection,1])
  ymin = min(coords[box_selection,0])
  ymax = max(coords[box_selection,0])

  # bbox = [xmin, xmax, ymin, ymax]
  bbox[0] = xmin
  bbox[1] = xmax
  bbox[2] = ymin
  bbox[3] = ymax

  logger.info(f'BBOX setting bbox for pulling from images: {bbox}')
  shared_variables['bbox'] = bbox # pretty sure this gets updated anyway

  # These are the messed up coordinates, to be used for plotting
  xmin_p = min(shared_variables['adata_data']['coordinates_1'][box_selection])
  xmax_p = max(shared_variables['adata_data']['coordinates_1'][box_selection])
  ymin_p = min(shared_variables['adata_data']['coordinates_2'][box_selection]) - shared_variables['y_shift']
  ymax_p = max(shared_variables['adata_data']['coordinates_2'][box_selection]) - shared_variables['y_shift']

  dx = xmax_p - xmin_p
  dy = ymax_p - ymin_p
  dimg = max(dx, dy)

  # Maintain aspect ratios defined by the actual coordinates
  figures['scatter_plot'].x_range.start = xmin_p
  figures['scatter_plot'].x_range.end = xmin_p + dimg
  figures['scatter_plot'].y_range.start = ymin_p# - shared_variables['y_shift']
  figures['scatter_plot'].y_range.end = ymin_p + dimg# - shared_variables['y_shift']

  shared_variables['bbox_plot'] = [xmin_p, xmax_p, ymin_p, ymax_p]

  # figures['image_plot'].x_range = figures['scatter_plot'].x_range
  # figures['image_plot'].y_range = figures['scatter_plot'].y_range

  # reset active images
  for c in use_channels:
    active_raw_images[c] = None