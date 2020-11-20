from os.path import dirname, join

import numpy as np
import scanpy as sc
import pandas as pd
# import pandas.io.sql as psql

from bokeh.io import curdoc
from bokeh.layouts import column, row, layout, gridplot
from bokeh.models import (ColumnDataSource, BoxSelectTool, LassoSelectTool, Button, 
                          Div, Select, Slider, TextInput, MultiChoice, ColorPicker, 
                          Dropdown, Span)
from bokeh.plotting import figure

import seaborn as sns
from matplotlib.colors import rgb2hex

from micron2.codexutils import get_images, blend_images
from .selection_ops import (logger, set_dropdown_menu, set_active_channel,
                            update_image_plot, update_bbox)

# from .selection_ops import logger as logg
# logger = make_logger()

ad = sc.read_h5ad("notebooks/tests/dataset.h5ad")
logger.info(f'Visualizing {ad.shape[0]} cells')
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
desc = Div(text=open(join(dirname(__file__), "description.html")).read(), sizing_mode="stretch_width")


## -------------------------------------------------------------------
#                      Shared variables
## -------------------------------------------------------------------

all_clusters = list(np.unique(ad.obs.mean_leiden))
all_channels = [k for k, i in ad.uns['image_sources'].items()]
active_raw_images = {c: None for c in all_channels}
saturation_vals = {c: 50 for c in all_channels} # TODO config default saturation values
use_files = [ad.uns['image_sources'][c] for c in ['DAPI']]

channel_colors = np.array(sns.color_palette('Set1', n_colors=len(all_channels)))
channel_colors = np.concatenate([channel_colors, np.ones((len(all_channels), 1))], axis=1)
channel_colors = {c: color for c, color in zip(all_channels, channel_colors)}

shared_variables = dict(
  all_clusters = all_clusters,
  all_channels = all_channels,
  bbox = [0,0,0,0],
  active_raw_images = active_raw_images,
  saturation_vals = saturation_vals,
  active_channel = 'DAPI',
  use_channels = ['DAPI'],
  channel_colors = channel_colors,
  nbins = 100,
  image_sources = ad.uns['image_sources'],
  background_color = '#636363',
  coords = coords
)

# Functions that pull data from somewhere and update data somewhere else
update_functions = {}


## -------------------------------------------------------------------
#               Set up a way to edit the displayed colors
## -------------------------------------------------------------------

## set up a color picker 
for c in all_channels:
  logger.info(f'{c} {rgb2hex(channel_colors[c])}')

widgets = dict(
  cluster_select = MultiChoice(title='Focus clusters', value=[], options=all_clusters),
  channels_select = MultiChoice(title='Show channels', value=['DAPI'], options=all_channels),

  ## Select and edit colors
  focus_channel = Dropdown(label='Edit DAPI image', button_type='primary', menu=['DAPI']),
  color_picker = ColorPicker(title='DAPI color', width=100, color=rgb2hex(channel_colors['DAPI'])),
  color_slider = Slider(start=0, end=255, step=1, title='DAPI saturation',                         
                        width=100, value=100, value_throttled=100),
  clear_clusters = Button(label='Clear focused cells', button_type='success'),
  clear_channels = Button(label='Clear image channels', button_type='success'),
  update_image = Button(label='Update drawing', button_type='success')
)



## -------------------------------------------------------------------
#                  Create data sources first
## -------------------------------------------------------------------

dummy_data = np.zeros((5,5), dtype=np.uint32)
figure_sources = dict(
  scatter_bg = ColumnDataSource(data=dict(x=[], y=[], mean_leiden=[], z_leiden=[])),
  scatter_fg = ColumnDataSource(data=dict(x=[], y=[], mean_leiden=[], z_leiden=[], color=[])),
  image_data = ColumnDataSource({'value': [], 'dw': [], 'dh': []}),
  cluster_hist = ColumnDataSource({'value': [0]*len(all_clusters), 'x': all_clusters}),
  intensity_hist = ColumnDataSource({'value': [], 'x': []})
)

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
width = 500
height = int(width * (dy/dx))
p = figure(plot_height=height, plot_width=width, title="", toolbar_location='left', 
           tools='pan,wheel_zoom,reset,hover,box_select,lasso_select,save', 
           sizing_mode="scale_both",
           tooltips=TOOLTIPS, 
           output_backend='webgl')
p.select(BoxSelectTool).select_every_mousemove = False
p.select(LassoSelectTool).select_every_mousemove = False

# Keep a reference to the foreground set
fg_scatter = p.scatter(x="x", y="y", source=figure_sources['scatter_fg'], 
              radius=12, color="active_color", line_color=None)
p.scatter(x="x", y="y", source=figure_sources['scatter_bg'], 
          radius=5, color=background_color, line_color=None)


## -------------------------------------------------------------------
#                  Create the zoom image figure
## -------------------------------------------------------------------
pimg = figure(plot_height=height, plot_width=int(width/2), title="", 
              toolbar_location='left', 
              tools='pan,wheel_zoom,reset,save', 
              match_aspect=True,
              output_backend='webgl')
pimg.image_rgba(image='value', source=figure_sources['image_data'],
                x=0,y=0,dw='dw',dh='dh')


## -------------------------------------------------------------------
#                  Create a selection histogram
## -------------------------------------------------------------------
phist = figure(x_range=all_clusters, 
               title="",
               x_axis_label='Clusters',
               y_axis_label='Cells',
               plot_height=height, 
               plot_width=int(width/2), 
               toolbar_location=None,
               output_backend='webgl')
phist.vbar(x='x', top='value',  source=figure_sources['cluster_hist'], width=0.9)
phist.xgrid.grid_line_color = None


## -------------------------------------------------------------------
#        Create a histogram of the image channel we're editing
## -------------------------------------------------------------------
pedit = figure(plot_height=200, plot_width=300, toolbar_location=None,
               output_backend='webgl', title='Intensity',
              #  x_axis_label='Intensity',
              #  y_axis_label='pixels'
              )
pedit_data_source = ColumnDataSource({'value': [0]*shared_variables['nbins'], 
                                      'x': range(shared_variables['nbins'])})
pedit.vbar(x='x', top='value', source=figure_sources['intensity_hist'], width=0.9,)


figures = dict(
  scatter_plot = p,
  image_plot = pimg,
  cluster_hist = phist,
  intensity_hist = pedit
)


## -------------------------------------------------------------------
#                      Set up the app layout
## -------------------------------------------------------------------
cluster_inputs = column([widgets['cluster_select'], 
                       widgets['clear_clusters']], 
                       width=300, height=200)
cluster_inputs.sizing_mode = "fixed"

image_inputs = column([widgets['channels_select'], 
                       widgets['clear_channels']], 
                       width=300, height=200)
image_inputs.sizing_mode = "fixed"

image_edit_inputs = column([widgets['focus_channel'], 
                            row(widgets['color_picker'], widgets['color_slider']),
                            widgets['update_image']], 
                            width=300, height=200)

inputs = row([cluster_inputs, 
              image_inputs, 
              image_edit_inputs, 
              figures['intensity_hist']], 
             height=300)

l = layout([[desc],
            [p],
            [inputs],
            [pimg, phist]
           ], 
           sizing_mode='scale_both',
)


## -------------------------------------------------------------------
#                      Image drawing functions
## -------------------------------------------------------------------

# Define actions for our buttons: 
# one main function per button that kicks off actions that should take 
# place when a value changes.


def handle_clear_clusters(*args):
  widgets['cluster_select'].value = []


def handle_clear_channels(*args):
  widgets['channels_select'].value = []


def select_cells():
    cluster_vals = widgets['cluster_select'].value
    if len(cluster_vals) == 0:
      idx = np.ones(data.shape[0], dtype='bool')
    else:
      idx = data.mean_leiden.isin(cluster_vals)

    fg_data = data[idx]
    bg_data = data[~idx]
    return fg_data, bg_data

def update_scatter():
    df_fg, df_bg = select_cells()

    figures['scatter_plot'].title.text = "Highlighting %d cells" % len(df_fg)
    figure_sources['scatter_fg'].data = dict(
        x=df_fg['coordinates_1'],
        y=df_fg['coordinates_2'],
        mean_leiden=df_fg['mean_leiden'],
        z_leiden=df_fg['z_leiden'],
        active_color=df_fg["active_color"],
    )
    figure_sources['scatter_bg'].data = dict(
        x=df_bg['coordinates_1'],
        y=df_bg['coordinates_2'],
        mean_leiden=df_bg['mean_leiden'],
        z_leiden=df_bg['z_leiden'],
    )


def update_clusters_hist(inds):
  selected_vals = ad.obs.mean_leiden.values[inds]
  vals = np.zeros(len(all_clusters), dtype=np.int)
  for i,u in enumerate(all_clusters):
    logger.info(f'selection cluster {u}: {(selected_vals == u).sum()}')
    vals[i] = (selected_vals == u).sum()
  figure_sources['cluster_hist'].data = {'value': vals, 'x': all_clusters}


def update_selection(attr, old, new):
  if len(new)==0:
    logger.info('Nothing selected')
    return
  inds = new
  update_bbox(inds, shared_variables)
  update_clusters_hist(inds)
  figures['scatter_plot'].title.text = "Highlighting %d cells" % len(inds)
  figures['cluster_hist'].title.text = "Highlighting %d cells" % len(inds)

# https://docs.bokeh.org/en/latest/docs/gallery/color_sliders.html
def hex_to_dec(hex):
  red = ''.join(hex.strip('#')[0:2])
  green = ''.join(hex.strip('#')[2:4])
  blue = ''.join(hex.strip('#')[4:6])
  return (int(red, 16), int(green, 16), int(blue,16), 255)


def update_color(attr, old, new):
  # Set the new active color for the channel in channel_colors and re-draw the image
  # active_channel = edit_image_dropdown.value
  color_hex = widgets['color_picker'].color
  color_rgb = hex_to_dec(color_hex)
  ac = shared_variables['active_channel']
  shared_variables['channel_colors'][ac] = np.array(color_rgb)/255.
  logger.info(f'new color: {color_hex} {color_rgb}')
  logger.info(f'set color for active_channel {ac} redrawing...')


def update_intensity(attr, old, new):
  ac = shared_variables['active_channel']
  print('setting channel', ac, 'saturation to', new)
  shared_variables['saturation_vals'][ac] = new


def handle_update_image(event):
  # update_image_plot(channels_multichoice, bbox, active_raw_images, channel_colors,
  #                     image_data_source, ad, saturation_vals)
  update_image_plot(shared_variables, widgets, figure_sources, figures)

def handle_set_active_channel(event):
  # This is a bear. lots of stuff happens when the active channel is changed.
  set_active_channel(event, shared_variables, widgets, figure_sources, figures)


# Pin action functions to controls
update_functions['handle_clear_clusters'] = handle_clear_clusters
update_functions['handle_clear_channels'] = handle_clear_channels
update_functions['update_scatter'] = update_scatter
update_functions['update_intensity'] = update_intensity
update_functions['update_color'] = update_color
update_functions['handle_focus_channel'] = handle_set_active_channel
update_functions['handle_update_image'] = handle_update_image


# Tie buttons to functions
widgets['cluster_select'].on_change('value', lambda attr, old, new: update_functions['update_scatter']())
widgets['channels_select'].on_change('value', lambda attr, old, new: set_dropdown_menu(shared_variables, widgets))
widgets['clear_clusters'].on_click(update_functions['handle_clear_clusters'])
widgets['clear_channels'].on_click(update_functions['handle_clear_channels'])
widgets['color_slider'].on_change('value_throttled', update_functions['update_intensity'])
widgets['color_picker'].on_change('color', update_functions['update_color'])
widgets['focus_channel'].on_click(update_functions['handle_focus_channel'])
widgets['update_image'].on_click(update_functions['handle_update_image'])

# Only select from foreground "in focus" cells
fg_scatter.data_source.selected.on_change('indices', update_selection)

logger.info('Populating initial data')
update_scatter()  # initial load of the data


logger.info('adding layout to root')
curdoc().add_root(l)
curdoc().title = "CODEX"
