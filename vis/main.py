from os.path import dirname, join

import numpy as np
import scanpy as sc
import pandas as pd
# import pandas.io.sql as psql

from bokeh.io import curdoc
from bokeh.layouts import column, row, layout, gridplot
from bokeh.models import (ColumnDataSource, BoxSelectTool, LassoSelectTool, Button, 
                          Div, Select, Slider, TextInput, MultiChoice, ColorPicker, 
                          Dropdown)
from bokeh.plotting import figure

import seaborn as sns
from matplotlib.colors import rgb2hex

from micron2.codexutils import get_images, blend_images


import logging
logger = logging.getLogger('leiden')
logger.setLevel('INFO')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

# figure and button definitions should live in other files
# this file is for linking them together and defining ops 

ad = sc.read_h5ad("notebooks/tests/dataset.h5ad")
print(f'\n\nVisualizing {ad.shape[0]} cells\n\n')
data = ad.obs.copy()
data[ad.var_names.tolist()] = pd.DataFrame(ad.X.copy(), index=ad.obs_names, columns=ad.var_names)
data['coordinates_1'] = ad.obsm['coordinates'][:,0]
data['coordinates_2'] = ad.obsm['coordinates'][:,1]

color_map = {k: v for k, v in zip(np.unique(ad.obs.mean_leiden), ad.uns['mean_leiden_colors'])}
data['color'] = [color_map[g] for g in ad.obs.mean_leiden]
data['active_color'] = [color_map[g] for g in ad.obs.mean_leiden]

# coords are stored with Y inverted for plotting with matplotlib.. flip it back for pulling from the images.
coords = ad.obsm['coordinates']
coords[:,1] = -coords[:,1]
# bbox_pad = 64 # minimum half-size to show when a single cell is selected

background_color = '#636363'

# axis_map = {
#   "coordinates_1": "coordinates_1",
#   "coordinates_2": "coordinates_2"
# }

desc = Div(text=open(join(dirname(__file__), "description.html")).read(), sizing_mode="stretch_width")

## Create Input controls
all_clusters = list(np.unique(ad.obs.mean_leiden))
hl_cluster_multichoice = MultiChoice(title='Focus clusters', value=[], options=all_clusters)

all_channels = [k for k, i in ad.uns['image_sources'].items()]
channels_multichoice = MultiChoice(title='Show channels', value=['DAPI'], options=all_channels)



## -------------------------------------------------------------------
#               Set up a way to edit the displayed colors
## -------------------------------------------------------------------
## Dropdown select button to choose active color
edit_image_dropdown = Dropdown(label='Edit DAPI image', button_type='primary', menu=['DAPI'])
def set_dropdown_menu():
  items = channels_multichoice.value
  edit_image_dropdown.menu = items


## set up a color picker 
channel_colors = np.array(sns.color_palette('Set1', n_colors=len(all_channels)))
channel_colors = np.concatenate([channel_colors, np.ones((len(all_channels), 1))], axis=1)
print(channel_colors.shape)
# default_channel_colors = np.concatenate([default_channel_colors, np.
channel_colors = {c: color for c, color in zip(all_channels, channel_colors)}
for c in all_channels:
  print(c, rgb2hex(channel_colors[c]))
image_colorpicker = ColorPicker(title='DAPI color', width=100, color=rgb2hex(channel_colors['DAPI']))


## set up an intensity slider
saturation_vals = {c: 255 for c in all_channels}
image_slider = Slider(start=0, end=255, value=255, step=1, title='DAPI saturation')



## 

# Create Column Data Source that will be used by the scatter plot
source_bg = ColumnDataSource(data=dict(x=[], y=[], mean_leiden=[], z_leiden=[]))
source_fg = ColumnDataSource(data=dict(x=[], y=[], mean_leiden=[], z_leiden=[], color=[]))


## -------------------------------------------------------------------
#                  Create the cell scatter figure
## -------------------------------------------------------------------

# Draw a canvas with an appropriate aspect ratio
TOOLTIPS=[
    ("Mean cluster", "@mean_leiden"),
    ("Morph cluster", "@z_leiden"),
]
dx = np.abs(data.coordinates_1.max() - data.coordinates_1.min())
dy = np.abs(data.coordinates_2.max() - data.coordinates_2.min())
width = 700
height = int(width * (dy/dx))
p = figure(plot_height=height, plot_width=width, title="", toolbar_location='left', 
           tools='pan,wheel_zoom,reset,hover,box_select,lasso_select,save', 
           sizing_mode="scale_both",
           tooltips=TOOLTIPS, 
           output_backend='webgl')
p.select(BoxSelectTool).select_every_mousemove = False
p.select(LassoSelectTool).select_every_mousemove = False

# Keep a reference to the foreground set
r = p.scatter(x="x", y="y", source=source_fg, radius=10, color="active_color", line_color=None)
p.scatter(x="x", y="y", source=source_bg, radius=5, color=background_color, line_color=None)
p.xaxis.axis_label = 'coordinates_1'
p.yaxis.axis_label = 'coordinates_2'


# Define actions for our buttons
def select_cells():
    hl_cluster_vals = hl_cluster_multichoice.value

    if len(hl_cluster_vals) == 0:
      hl_idx = np.ones(data.shape[0], dtype='bool')
    else:
      hl_idx = data.mean_leiden.isin(hl_cluster_vals)

    hl_data = data[hl_idx]
    bg_data = data[~hl_idx]

    return hl_data, bg_data


def update():
    df_fg, df_bg = select_cells()

    p.title.text = "Highlighting %d cells" % len(df_fg)
    source_fg.data = dict(
        x=df_fg['coordinates_1'],
        y=df_fg['coordinates_2'],
        mean_leiden=df_fg['mean_leiden'],
        z_leiden=df_fg['z_leiden'],
        active_color=df_fg["active_color"],
    )
    source_bg.data = dict(
        x=df_bg['coordinates_1'],
        y=df_bg['coordinates_2'],
        mean_leiden=df_bg['mean_leiden'],
        z_leiden=df_bg['z_leiden'],
    )


# Pin action functions to controls
# Controls that will update the plotting area
hl_cluster_multichoice.on_change('value', lambda attr, old, new: update())



# Add a clear button for multiselects
def set_cleared_focus(*args):
  hl_cluster_multichoice.value = []
clear_button_focus = Button(label='Clear focused cells', button_type='success')
clear_button_focus.on_click(set_cleared_focus)

def set_cleared_images(*args):
  channels_multichoice.value = []
clear_button_image = Button(label='Clear image channels', button_type='success')
clear_button_image.on_click(set_cleared_images)



## -------------------------------------------------------------------
#                  Create the zoom image figure
## -------------------------------------------------------------------
pimg = figure(plot_height=height, plot_width=int(width/2), title="", toolbar_location='left', 
              tools='pan,wheel_zoom,reset,save', 
              # sizing_mode="scale_both",
              # tooltips=TOOLTIPS, 
              output_backend='webgl')
img_data = np.random.randint(low=0, high=2**32, size=(512 , 512), dtype=np.uint32)*0
image_data_source = ColumnDataSource({'value': [img_data]})
pimg.image_rgba(image='value', source=image_data_source,x=0,y=0,dw=1,dh=1)




## -------------------------------------------------------------------
#                  Create a selection histogram
## -------------------------------------------------------------------
phist = figure(x_range=all_clusters, plot_height=height, plot_width=int(width/2), toolbar_location=None,
               output_backend='webgl')

phist_data_source = ColumnDataSource({'value': [0]*len(all_clusters), 'x': all_clusters})
phist.vbar(x='x', top='value', source=phist_data_source, width=0.9)
phist.xgrid.grid_line_color = None
# phist.y_range_start = 0


## -------------------------------------------------------------------
#        Create a histogram of the image channel we're editing
## -------------------------------------------------------------------
nbins = 100
pedit = figure(plot_height=200, plot_width=200, toolbar_location=None,
               output_backend='webgl')
pedit_data_source = ColumnDataSource({'value': [0]*nbins, 'x': range(nbins)})
pedit.vbar(x='x', top='value', source=pedit_data_source, width=1)
# pedit.xgrid.grid_line_color = None
# phist.y_range_start = 0



## -------------------------------------------------------------------
#                      Set up the app layout
## -------------------------------------------------------------------
focus_inputs = column([hl_cluster_multichoice, clear_button_focus], width=300, height=200)
focus_inputs.sizing_mode = "fixed"
image_inputs = column([channels_multichoice, clear_button_image], width=300, height=200)
image_inputs.sizing_mode = "fixed"

image_edit_inputs = column([edit_image_dropdown, 
                            row(image_colorpicker, image_slider)], 
                            width=300, height=200)

inputs = row([focus_inputs, 
              image_inputs, 
              image_edit_inputs, 
              pedit], 
             height=200)
# inputs.sizing_mode = "fixed"

l = layout([[desc],
            [p],
            [inputs],
            [pimg, phist]
           ], 
           sizing_mode='scale_both',
)

print('Populating initial data')
update()  # initial load of the data


## -------------------------------------------------------------------
#                      Image drawing functions
## -------------------------------------------------------------------
bbox = [0, 0, 0, 0]
active_raw_images = {c: None for c in all_channels}
def update_bbox(inds):
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
  use_channels = channels_multichoice.value
  for c in use_channels:
    active_raw_images[c] = None


# set up a persistent variable to hold the active image for each channel
def update_image_plot():
  use_channels = channels_multichoice.value
  if len(use_channels) == 0:
    print('No channels selected')
    return

  if sum(bbox) == 0:
    print('No bbox to draw')
    return

  ## Decide if we have to read images from disk or not
  need_to_pull = False
  for c in use_channels:
    img = active_raw_images[c]
    if img is None:
      need_to_pull = True
      break
        
  if need_to_pull:
    print(f'Pulling bbox: {bbox} from {use_channels}')
    use_files = [ad.uns['image_sources'][c] for c in use_channels]
    images = get_images(use_files, bbox)
  else:
    images = np.dstack([active_raw_images[c] for c in use_channels])

  for i, c in enumerate(use_channels):
    active_raw_images[c] = images[:,:,i].copy()

  colors = []
  for c in use_channels:
    chc = (np.array(channel_colors[c])*255).astype(np.uint8)
    print('setting color', c, chc)
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
  image_data_source.data = {'value': [blended]}


def update_clusters_hist(inds):
  selected_vals = ad.obs.mean_leiden.values[inds]
  vals = np.zeros(len(all_clusters), dtype=np.int)
  for i,u in enumerate(all_clusters):
    print(f'selection cluster {u}: {(selected_vals == u).sum()}')
    vals[i] = (selected_vals == u).sum()
  phist_data_source.data = {'value': vals, 'x': all_clusters}


def update_selection(attr, old, new):
  if len(new)==0:
    # selected nothing, take no action
    return

  print(attr)

  inds = new
  update_bbox(inds)
  update_clusters_hist(inds)
  update_image_plot()
  p.title.text = "Highlighting %d cells" % len(inds)


def update_edit_hist(pedit_data_source):
  img = active_raw_images[active_channel[0]].ravel()
  hist, edges = np.histogram(img, bins=nbins)
  pedit_data_source = {'value': hist, 'x': edges}


def channels_changed(attr, old, new):
  set_dropdown_menu()
  update_image_plot()


# https://docs.bokeh.org/en/latest/docs/gallery/color_sliders.html
def hex_to_dec(hex):
  red = ''.join(hex.strip('#')[0:2])
  green = ''.join(hex.strip('#')[2:4])
  blue = ''.join(hex.strip('#')[4:6])
  return (int(red, 16), int(green, 16), int(blue,16), 255)


def update_color(attr, old, new):
  # Set the new active color for the channel in channel_colors and re-draw the image
  # active_channel = edit_image_dropdown.value
  color_hex = image_colorpicker.color
  color_rgb = hex_to_dec(color_hex)
  channel_colors[active_channel[0]] = np.array(color_rgb)/255.
  print('new color:', color_hex, color_rgb)
  print('set color for active_channel', active_channel[0], 'redrawing...')
  update_image_plot()


def update_intensity(attr, old, new):
  print('setting channel', active_channel[0], 'saturation to', new)
  saturation_vals[active_channel[0]] = new
  update_image_plot()



# Now wire the editing options together
active_channel = ['DAPI']
def set_active_channel(event):
  active_channel[0] = event.item
  print('setting active_channel', active_channel)
  edit_image_dropdown.label = f'Edit {active_channel[0]} image'
  image_colorpicker.title = f'{active_channel[0]} color'
  image_colorpicker.color = rgb2hex(channel_colors[active_channel[0]])
  image_slider.title = f'{active_channel[0]} saturation'
  image_slider.value = saturation_vals[active_channel[0]]
  update_edit_hist(pedit_data_source)

edit_image_dropdown.on_click(set_active_channel)



# slider needs to be on release or something..
image_slider.on_change('value', update_intensity)
image_colorpicker.on_change('color', update_color)
channels_multichoice.on_change('value', channels_changed)

# Only select from foreground "in focus" cells
r.data_source.selected.on_change('indices', update_selection)

curdoc().add_root(l)
curdoc().title = "CODEX"
