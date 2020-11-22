from os.path import dirname, join

import numpy as np
import scanpy as sc
import pandas as pd
# import pandas.io.sql as psql

from bokeh.io import curdoc
from bokeh.layouts import column, row, layout, gridplot
from bokeh.models import (ColumnDataSource, BoxSelectTool, LassoSelectTool, Button, 
                          Div, Select, Slider, TextInput, MultiChoice, MultiSelect, ColorPicker, 
                          Dropdown, Span, CheckboxButtonGroup, 
                          CheckboxGroup, Spinner)
from bokeh.plotting import figure

import seaborn as sns
from matplotlib.colors import rgb2hex

from bokeh.themes import built_in_themes

from micron2.codexutils import get_images, blend_images
from micron2.spatial import get_neighbors, pull_neighbors
from .selection_ops import (logger, set_dropdown_menu, set_active_channel,
                            update_image_plot, update_bbox)
from .scatter_gate import ScatterGate


# curdoc().theme = 'dark_minimal'
# from bokeh.themes import Theme
# curdoc().theme = Theme(filename='theme.yaml')

# from .selection_ops import logger as logg
# logger = make_logger()

ad = sc.read_h5ad("notebooks/tests/dataset.h5ad")

logger.info(f'Visualizing {ad.shape[0]} cells')
data = ad.obs.copy()
data[ad.var_names.tolist()] = pd.DataFrame(ad.X.copy(), index=ad.obs_names, columns=ad.var_names)
data['index_num'] = np.arange(data.shape[0])
data['coordinates_1'] = ad.obsm['coordinates'][:,0]
data['coordinates_2'] = ad.obsm['coordinates'][:,1]

color_map = {k: v for k, v in zip(np.unique(ad.obs.mean_leiden), ad.uns['mean_leiden_colors'])}
data['color'] = [color_map[g] for g in ad.obs.mean_leiden]
# data['active_color'] = [color_map[g] for g in ad.obs.mean_leiden]

# coords are stored with Y inverted for plotting with matplotlib.. flip it back for pulling from the images.
coords = ad.obsm['coordinates']
coords[:,1] = -coords[:,1]
# bbox_pad = 64 # minimum half-size to show when a single cell is selected


background_color = '#636363'
desc = Div(text=open(join(dirname(__file__), "description.html")).read(), sizing_mode="stretch_width")


## -------------------------------------------------------------------
#                      Shared variables
## -------------------------------------------------------------------

clusters = np.array(ad.obs.mean_leiden)
all_clusters = list(np.unique(clusters))
all_channels = [k for k, i in ad.uns['image_sources'].items()]
active_raw_images = {c: None for c in all_channels}
saturation_vals = {c: 50 for c in all_channels} # TODO config default saturation values
use_files = [ad.uns['image_sources'][c] for c in ['DAPI']]
neighbor_indices = get_neighbors(coords)

channel_colors = np.array(sns.color_palette('Set1', n_colors=len(all_channels)))
channel_colors = np.concatenate([channel_colors, np.ones((len(all_channels), 1))], axis=1)
channel_colors = {c: color for c, color in zip(all_channels, channel_colors)}

shared_variables = dict(
  clusters = clusters,
  n_cells = len(clusters),
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
  foreground_color = '#e34a33',
  neighbor_color = '#5e6bf2',
  coords = coords,
  neighbors = neighbor_indices,
  box_selection = np.ones(coords.shape[0], dtype=bool),
  cluster_selection = np.ones(coords.shape[0], dtype=bool),
  highlight_cells = np.zeros(coords.shape[0], dtype=bool),
  neighbor_cells = np.zeros(coords.shape[0], dtype=bool)
)

# Functions that pull data from somewhere and update data somewhere else
update_functions = {}


## -------------------------------------------------------------------
#               Set up a way to edit the displayed colors
## -------------------------------------------------------------------

## set up a color picker 
for c in all_channels:
  logger.info(f'{c} {rgb2hex(channel_colors[c])}')

CLUSTER_VIEW_OPTS = [
  'Join focused clusters',
  'Show neighbors'
]
widgets = dict(
  cluster_select = MultiSelect(title='Focus clusters', value=[], options=all_clusters, 
                               height=200,
                               css_classes=["my-widgets"]),
  channels_select = MultiSelect(title='Show channels', value=['DAPI'], options=all_channels,
                                height=200,
                                css_classes=["my-widgets"]),

  ## Options for selecting cells via cluster
  cluster_view_opts = CheckboxGroup(labels=CLUSTER_VIEW_OPTS, active=[],css_classes=["my-widgets"]),

  ## Select and edit colors
  focus_channel = Dropdown(label='Edit DAPI image', button_type='primary', menu=['DAPI'],),
  color_picker = ColorPicker(title='DAPI color', width=100, color=rgb2hex(channel_colors['DAPI']),
                             css_classes=["my-widgets"]),
  color_saturation = Spinner(low=0, high=255, step=1, title='DAPI saturation',                         
                             width=80, value=50, css_classes=["my-widgets"]),
  clear_clusters = Button(label='Clear focused cells', button_type='success'),
  clear_channels = Button(label='Clear image channels', button_type='success'),
  update_image = Button(label='Update drawing', button_type='success')
)

# widgets['cluster_select'].title.text_color = '#e7e7e7'
# widgets['channels_select'].title.text_color = '#e7e7e7'


## -------------------------------------------------------------------
#                  Create data sources first
## -------------------------------------------------------------------

dummy_data = np.zeros((5,5), dtype=np.uint32)
figure_sources = dict(
  # scatter_bg = ColumnDataSource(data=dict(x=[], y=[], index=[], mean_leiden=[], z_leiden=[])),
  scatter_fg = ColumnDataSource(data=dict(x=[], y=[], s=[], index=[], mean_leiden=[], z_leiden=[], color=[])),
  image_data = ColumnDataSource({'value': [], 'dw': [], 'dh': []}),
  cluster_hist = ColumnDataSource({'value': [0]*len(all_clusters), 'x': all_clusters}),
  neighbor_hist = ColumnDataSource({'value': [0]*len(all_clusters), 'x': all_clusters}),
  intensity_hist = ColumnDataSource({'value': [], 'x': []})
)

## -------------------------------------------------------------------
#                  Create the cell scatter figure
## -------------------------------------------------------------------

# Draw a canvas with an appropriate aspect ratio
TOOLTIPS=[
    ("Index", "@index"),
    ("Mean cluster", "@mean_leiden"),
    ("Morph cluster", "@z_leiden"),
]
dx = np.abs(data.coordinates_1.max() - data.coordinates_1.min())
dy = np.abs(data.coordinates_2.max() - data.coordinates_2.min())
width = 400
# height = int(width * (dy/dx))
height = 400
p = figure(plot_height=height, plot_width=width, title="", toolbar_location='left', 
           tools='pan,wheel_zoom,reset,hover,box_select,lasso_select,save', 
           sizing_mode="scale_both",
           match_aspect=True,
           tooltips=TOOLTIPS, 
           output_backend='webgl')
p.select(BoxSelectTool).select_every_mousemove = False
p.select(LassoSelectTool).select_every_mousemove = False
fg_scatter = p.scatter(x="x", y="y", source=figure_sources['scatter_fg'], 
              radius='s', color="active_color", line_color=None)


## -------------------------------------------------------------------
#                  Create the zoom image figure
## -------------------------------------------------------------------
pimg = figure(plot_height=height, plot_width=width, title="", 
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
               plot_height=int(width/2), 
               toolbar_location=None,
               output_backend='webgl')
phist.vbar(x='x', top='value',  source=figure_sources['cluster_hist'], width=0.9)
phist.xgrid.grid_line_color = None
phist.xaxis.axis_label_text_font_size = '16pt'
phist.yaxis.axis_label_text_font_size = '16pt'
phist.xaxis.major_label_text_font_size = '12pt'
phist.yaxis.major_label_text_font_size = '12pt'

pnhist = figure(x_range=all_clusters, 
               title="",
               x_axis_label='Clusters',
               y_axis_label='Cells',
               plot_height=int(width/2), 
               toolbar_location=None,
               output_backend='webgl')
pnhist.vbar(x='x', top='value',  source=figure_sources['neighbor_hist'], width=0.9)
pnhist.xgrid.grid_line_color = None
pnhist.xaxis.axis_label_text_font_size = '16pt'
pnhist.yaxis.axis_label_text_font_size = '16pt'
pnhist.xaxis.major_label_text_font_size = '12pt'
pnhist.yaxis.major_label_text_font_size = '12pt'

## -------------------------------------------------------------------
#        Create a histogram of the image channel we're editing
## -------------------------------------------------------------------
pedit = figure(plot_height=200, plot_width=300, toolbar_location=None,
               output_backend='webgl', title='Intensity',
               x_axis_label='Intensity',
               y_axis_label='pixels (log)'
              )
pedit_data_source = ColumnDataSource({'value': [0]*shared_variables['nbins'], 
                                      'x': range(shared_variables['nbins'])})
pedit.vbar(x='x', top='value', source=figure_sources['intensity_hist'], width=0.9,)
pedit.xaxis.axis_label_text_font_size = '10pt'
pedit.yaxis.axis_label_text_font_size = '10pt'
pedit.xaxis.major_label_text_font_size = '8pt'
pedit.yaxis.major_label_text_font_size = '8pt'



## -------------------------------------------------------------------
#                      Selection scatter plot
## -------------------------------------------------------------------


scatter_gate_module = ScatterGate(data[all_channels], 
                                  height = int(width/2),
                                  width =  int(width/2),
                                  initial_values = all_channels[1:3])



figures = dict(
  scatter_plot = p,
  image_plot = pimg,
  cluster_hist = phist,
  neighbor_hist = pnhist,
  intensity_hist = pedit,
  scatter_gate = scatter_gate_module
)


## -------------------------------------------------------------------
#                      Set up the app layout
## -------------------------------------------------------------------
cluster_inputs = column([widgets['cluster_select'], 
                         widgets['clear_clusters'], 
                         widgets['cluster_view_opts'],],
                         width=300, height=300
                       )
# cluster_inputs.sizing_mode = "fixed"

image_inputs = column(
  widgets['channels_select'], 
  widgets['clear_channels'], 
  width=300, height=300
)
# image_inputs.sizing_mode = "fixed"

image_edit_inputs = column([widgets['focus_channel'], 
                            row(widgets['color_picker'], widgets['color_saturation']),
                            widgets['update_image'],
                            figures['intensity_hist']], 
                            width=300, height=200)

left_inputs = column([
  cluster_inputs, 
  image_inputs, 
  image_edit_inputs,
  ], 
  width=300, height=1000)
left_inputs.sizing_mode = 'fixed'
            

right_inputs = column([
  figures['scatter_gate'].selection,
  figures['scatter_gate'].sliders,
  ], 
  width=300, height=1000)
right_inputs.sizing_mode = 'fixed'

selection_hists = row(
  figures['cluster_hist'], 
  figures['neighbor_hist'],
  sizing_mode='scale_both'
)


all_plots = layout([
  [figures['scatter_plot'], figures['image_plot'], figures['scatter_gate'].FIG], 
  [selection_hists] 
], sizing_mode='scale_both')


data_layout = layout([ 
  [left_inputs, all_plots, right_inputs],
], sizing_mode='scale_both')


l = layout([[desc], [data_layout]], 
           sizing_mode='scale_both',
)


## -------------------------------------------------------------------
#                      Image drawing functions
## -------------------------------------------------------------------

# Define actions for our buttons: 
# one main function per button that kicks off actions that should take 
# place when a value changes.


def handle_clear_clusters(*args):
  widgets['cluster_select'].value = [] # triggers update scatter


def handle_clear_channels(*args):
  widgets['channels_select'].value = []


def get_gated_cells():
  gate_selection_idx = figures['scatter_gate'].data.index[figures['scatter_gate'].selected]
  gate_selection = data.index.isin(gate_selection_idx)
  


def get_selected_clusters():
  cluster_vals = widgets['cluster_select'].value
  logger.info(f'setting selected clusters to: {cluster_vals}')

  cluster_vect = shared_variables['clusters']

  cluster_selection = np.zeros(shared_variables['n_cells'], dtype=bool)
  # nothing selected --> everything selected
  if len(cluster_vals) == 0:
    cluster_selection[:] = 1
  else:
    neighbors = []
    for v in cluster_vals:
      cluster_selection[shared_variables['clusters'] == v] = 1
      neighbors.append(pull_neighbors(shared_variables['neighbors'], cluster_vect, v))
    neighbors = np.sum(neighbors, axis=0) > 0
    # Remove focused clusters from the neighbors
    neighbors = neighbors & ~cluster_selection
    logger.info(f'pulled {neighbors.sum()} neighbor cells')
    shared_variables['neighbor_cells'] = neighbors & shared_variables['box_selection']

  logger.info(f'cluster selection: {cluster_selection.sum()}')
  shared_variables['cluster_selection'] = cluster_selection
  # Also update highlight cells
  shared_variables['highlight_cells'] = shared_variables['box_selection'] & \
                                        shared_variables['cluster_selection']


def update_neighbors_hist():
  idx = shared_variables['neighbor_cells']
  selected_vals = shared_variables['clusters'][idx]

  vals = np.zeros(len(shared_variables['all_clusters']), dtype=np.int)
  for i,u in enumerate(shared_variables['all_clusters']):
    logger.info(f'selection cluster {u}: {(selected_vals == u).sum()}')
    vals[i] = (selected_vals == u).sum()
  figure_sources['neighbor_hist'].data = {'value': vals, 'x': shared_variables['all_clusters']}


def update_clusters_hist():
  # Intersect focused clusters and box selection
  get_selected_clusters() 
  idx = shared_variables['cluster_selection'] & shared_variables['box_selection']
  selected_vals = shared_variables['clusters'][idx]

  vals = np.zeros(len(shared_variables['all_clusters']), dtype=np.int)
  for i,u in enumerate(shared_variables['all_clusters']):
    logger.info(f'selection cluster {u}: {(selected_vals == u).sum()}')
    vals[i] = (selected_vals == u).sum()
  figure_sources['cluster_hist'].data = {'value': vals, 'x': shared_variables['all_clusters']}


def update_box_selection(attr, old, new):
  if len(new)==0:
    box_selection = np.ones(shared_variables['coords'].shape[0], dtype=bool)
  else:
    inds = new
    box_selection = np.zeros_like(shared_variables['box_selection'], dtype=bool)
    box_selection[inds] = 1

  shared_variables['box_selection'] = box_selection
  logger.info(f'bbox selection: {box_selection.sum()}')

  shared_variables['highlight_cells'] = box_selection & shared_variables['cluster_selection']

  n_hl = shared_variables['highlight_cells'].sum()
  n_nbr = shared_variables['neighbor_cells'].sum()
  logger.info(f'total highlight cells: {n_hl}')

  update_bbox(shared_variables)
  update_clusters_hist()
  update_neighbors_hist()
  update_scatter()
  figures['scatter_plot'].title.text = "Highlighting %d cells" % n_hl
  figures['cluster_hist'].title.text = "Highlighting %d cells" % n_hl
  figures['neighbor_hist'].title.text = "%d neighboring cells" % n_nbr


def update_scatter():
  get_selected_clusters()
  n_hl = shared_variables['highlight_cells'].sum()
  n_nbr = shared_variables['neighbor_cells'].sum()

  # Update the gate scatter plot 
  figures['scatter_gate'].update_data(data.loc[shared_variables['highlight_cells']])

  logger.info(f'updating scatter with options: {widgets["cluster_view_opts"].active}')
  logger.info(f'highlighting total {n_hl} cells')

  if 0 in widgets['cluster_view_opts'].active:
    fg_colors = np.array([shared_variables['foreground_color']] * data.shape[0])
  else:
    fg_colors = np.array(data['color'])

  # in_box_not_selected = shared_variables['box_selection'] & ~shared_variables['cluster_selection']
  fg_colors[~shared_variables['highlight_cells']] = shared_variables['background_color']

  sizes = np.zeros(data.shape[0])+8
  if 1 in widgets['cluster_view_opts'].active:
    logger.info('coloring neighbors')
    fg_colors[shared_variables['neighbor_cells']] = shared_variables['neighbor_color']
    sizes[~shared_variables['highlight_cells'] & ~shared_variables['neighbor_cells']] = 5
  else:
    sizes[~shared_variables['highlight_cells']] = 5

  figures['scatter_plot'].title.text = "Highlighting %d cells" % n_hl
  figures['cluster_hist'].title.text = "Highlighting %d cells" % n_hl
  figures['neighbor_hist'].title.text = "%d neighboring cells" % n_nbr

  figure_sources['scatter_fg'].data = dict(
    x=data['coordinates_1'],
    y=data['coordinates_2'],
    s=sizes,
    index=data['index_num'],
    mean_leiden=data['mean_leiden'],
    z_leiden=data['z_leiden'],
    active_color=fg_colors,
  )
  update_clusters_hist()
  update_neighbors_hist()


# https://docs.bokeh.org/en/latest/docs/gallery/color_sliders.html
def hex_to_dec(hex):
  red = ''.join(hex.strip('#')[0:2])
  green = ''.join(hex.strip('#')[2:4])
  blue = ''.join(hex.strip('#')[4:6])
  return (int(red, 16), int(green, 16), int(blue,16), 255)


def update_color(attr, old, new):
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
  update_image_plot(shared_variables, widgets, figure_sources, figures)


# This is a bear. lots of stuff happens when the active channel is changed.
def handle_set_active_channel(event):
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
widgets['cluster_view_opts'].on_click(lambda event: update_functions['update_scatter']())
widgets['channels_select'].on_change('value', lambda attr, old, new: set_dropdown_menu(shared_variables, widgets))
widgets['clear_clusters'].on_click(update_functions['handle_clear_clusters'])
widgets['clear_channels'].on_click(update_functions['handle_clear_channels'])
widgets['color_saturation'].on_change('value', update_functions['update_intensity'])
widgets['color_picker'].on_change('color', update_functions['update_color'])
widgets['focus_channel'].on_click(update_functions['handle_focus_channel'])
widgets['update_image'].on_click(update_functions['handle_update_image'])


# Only select from foreground "in focus" cells
fg_scatter.data_source.selected.on_change('indices', update_box_selection)


logger.info('Populating initial data')
update_scatter()  # initial load of the data


logger.info('adding layout to root')
curdoc().add_root(l)
curdoc().title = "CODEX"
