from os.path import dirname, join

import numpy as np
import scanpy as sc
import pandas as pd
import cv2
# import pandas.io.sql as psql

from bokeh.io import curdoc
from bokeh.layouts import column, row, layout, gridplot
from bokeh.models import (ColumnDataSource, BoxSelectTool, LassoSelectTool, Button, 
                          Div, Select, Slider, TextInput, MultiChoice, MultiSelect, ColorPicker, 
                          Dropdown, Span, CheckboxButtonGroup, Toggle, FileInput,
                          CheckboxGroup, Spinner)
from bokeh.plotting import figure
from bokeh.models import Range1d

import glob
import seaborn as sns
from matplotlib.colors import rgb2hex
from scipy.sparse import issparse

# from bokeh.themes import built_in_themes
# from micron2.codexutils import get_images, blend_images
from micron2.spatial import pull_neighbors

from .selection_ops import (logger, set_dropdown_menu, 
                            update_image_plot, update_bbox)
from .load_data import set_active_slide

# from .scatter_gate import ScatterGate
from .boxplot import BokehBoxPlot

# curdoc().theme = 'dark_minimal'
# from bokeh.themes import Theme
# curdoc().theme = Theme(filename='theme.yaml')

# from .selection_ops import logger as logg
# logger = make_logger()

# data_dir = '/home/ingn/tmp/micron2-data/preprocessed_data'
# sample_id = '201021_BreastFFPE_Final'
# default_annotation = 'annotation'

#----------------------------------------------------------- #
#                      Input data                            #
#----------------------------------------------------------- #

# I need here a dummy file, or to load from a config file, 
# instead of having a default hard-coded.
# The problem is setting up all the widgets requires data
# that i pull in from the active file.
data_dir = '/Users/ingn/projects/codex-data/'
full_sample_id = f'210113_Breast_Cassette11_reg1'
default_annotation = 'annotation_main_features'

shared_variables = {}
active_file = f'{data_dir}/{full_sample_id}/{full_sample_id}.h5ad'
set_active_slide(active_file, shared_variables, default_annotation, logger)

# # ----------------------------

background_color = '#636363'
desc = Div(text=open(join(dirname(__file__), "description.html")).read(), sizing_mode="stretch_width")

# Functions that pull data from somewhere and update data somewhere else
update_functions = {}

## -------------------------------------------------------------------
#               Set up a way to edit the displayed colors
## -------------------------------------------------------------------

## set up a color picker 
for c in shared_variables['all_channels']:
  logger.info(f'channel {c} {rgb2hex(shared_variables["channel_colors"][c])}')

CLUSTER_VIEW_OPTS = [
  'Join focused clusters',
  'Show neighbors'
]
widgets = dict(
  cluster_select = CheckboxGroup(active=[], labels=shared_variables['all_clusters'], 
                                #  height=200, 
                                 margin=(10,10,10,10),
                                 css_classes=["my-widgets"]),
  channels_select = MultiSelect(title='Show channels', value=['DAPI'], options=shared_variables['all_channels'],
                                height=200, margin=(10,10,10,10),
                                css_classes=["my-widgets"]),

  ## Options for selecting cells via cluster
  cluster_view_opts = CheckboxGroup(labels=CLUSTER_VIEW_OPTS, active=[],
                                    margin=(10,10,10,10),
                                    css_classes=["my-widgets"]),
  clear_clusters = Button(label='Clear focused cells', button_type='success'),
  clear_channels = Button(label='Clear image channels', button_type='success'),
  update_image = Button(label='Update drawing', button_type='success'),
  # data_home = TextInput(name='Data home', placeholder='Enter full path to data home'),
  input_file = TextInput(placeholder='Enter full path to sample h5ad')
)

## Add selector for showing nuclei & nuclei color
widgets[f'focus_channel_nuclei'] = Toggle(label='nuclei', button_type='success', width=75)
widgets[f'color_picker_nuclei'] = ColorPicker(width=50, color='#f2e70a', css_classes=["my-widgets"])

## Add selectors for showing each channel & their color
for ch in shared_variables['all_channels']: 
  widgets[f'focus_channel_{ch}'] = Toggle(label=ch, button_type='success', width=75)
  widgets[f'color_picker_{ch}'] = ColorPicker(width=50, color=rgb2hex(shared_variables['channel_colors'][ch]),
                                              css_classes=["my-widgets"])
  widgets[f'color_saturation_{ch}'] = Spinner(low=0, high=255, step=1, tags=[ch], #title=f'{ch} saturation',           
                                              width=70, value=0, css_classes=["my-widgets"])


## -------------------------------------------------------------------
#                  Create data sources first
## -------------------------------------------------------------------

dummy_data = np.zeros((5,5), dtype=np.uint32)
figure_sources = dict(
  scatter_fg = ColumnDataSource(data=dict(x=[], y=[], s=[], index=[], annotation=[], color=[])),
  image_data = ColumnDataSource({'value': [], 'dw': [], 'dh': [], 'x0': [], 'y0': []}),
  cluster_hist = ColumnDataSource({'value':  [0]*len(shared_variables['all_clusters']), 'x': shared_variables['all_clusters']}),
  neighbor_hist = ColumnDataSource({'value': [0]*len(shared_variables['all_clusters']), 'x': shared_variables['all_clusters']}),
  intensity_hist = ColumnDataSource({'value': [], 'x': []})
)

## -------------------------------------------------------------------
#                  Create the cell scatter figure
## -------------------------------------------------------------------

# Draw a canvas with an appropriate aspect ratio
TOOLTIPS=[
    ("Index", "@index"),
    ("Mean cluster", "@annotation"),
    ("x", "@x"),
    ("y", "@y"),
]
# dx = np.abs(shared_variables['adata_data'].coordinates_1.max() - shared_variables['adata_data'].coordinates_1.min())
# dy = np.abs(shared_variables['adata_data'].coordinates_2.max() - shared_variables['adata_data'].coordinates_2.min())
width = 400
height = 400
p = figure(plot_height=height, plot_width=width, title="", toolbar_location='left', 
           x_range=Range1d(), y_range=Range1d(),
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
              x_range=p.x_range,
              y_range=p.y_range,
              toolbar_location='left', 
              tools='pan,wheel_zoom,reset,save', 
              match_aspect=True,
              output_backend='webgl')
pimg.image_rgba(image='value', source=figure_sources['image_data'],
                x='x0',y='y0',dw='dw',dh='dh')


## -------------------------------------------------------------------
#                  Create a selection histogram
## -------------------------------------------------------------------
phist = figure(x_range=shared_variables['all_clusters'], 
               title="",
               x_axis_label='Clusters',
               y_axis_label='Cells',
               plot_height=int(height*0.7), 
               toolbar_location=None,
               output_backend='webgl')
phist.vbar(x='x', top='value',  source=figure_sources['cluster_hist'], width=0.9)
phist.xgrid.grid_line_color = None
phist.xaxis.axis_label_text_font_size = '12pt'
phist.yaxis.axis_label_text_font_size = '12pt'
phist.xaxis.major_label_text_font_size = '8pt'
phist.yaxis.major_label_text_font_size = '10pt'
phist.xaxis.major_label_orientation = np.pi/2

pnhist = figure(x_range=shared_variables['all_clusters'], 
               title="",
               x_axis_label='Clusters',
               y_axis_label='Cells',
               plot_height=int(height*0.7), 
               toolbar_location=None,
               output_backend='webgl')
pnhist.vbar(x='x', top='value',  source=figure_sources['neighbor_hist'], width=0.9)
pnhist.xgrid.grid_line_color = None
pnhist.xaxis.axis_label_text_font_size = '12pt'
pnhist.yaxis.axis_label_text_font_size = '12pt'
pnhist.xaxis.major_label_text_font_size = '8pt'
pnhist.yaxis.major_label_text_font_size = '10pt'
pnhist.xaxis.major_label_orientation = np.pi/2



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
# scatter_gate_module = ScatterGate(data[all_channels], 
#                                   height = int(width/2),
#                                   width =  int(width/2),
#                                   initial_values = all_channels[1:3])


## -------------------------------------------------------------------
#                           Marker boxplot
## -------------------------------------------------------------------

boxplot_module = BokehBoxPlot(shared_variables['adata_data'], 
                              varnames=[v for v in shared_variables['all_channels'] if v !='DAPI'],
                              height=200, width=1000)

figures = dict(
  scatter_plot = p,
  image_plot = pimg,
  cluster_hist = phist,
  neighbor_hist = pnhist,
  intensity_hist = pedit,
  # scatter_gate = scatter_gate_module,
  boxplot = boxplot_module
)

## -------------------------------------------------------------------
#                      Set up the app layout
## -------------------------------------------------------------------
left_inputs = column([
  widgets['input_file'],
  widgets['cluster_select'], 
  widgets['clear_clusters'], 
  widgets['cluster_view_opts'],],
  margin=(10,10,10,10),
  width=200)
left_inputs.sizing_mode = 'stretch_height'
            
# Build the channel colors
color_inputs = [row([widgets['focus_channel_nuclei'], widgets['color_picker_nuclei']])]
for ch in shared_variables['all_channels']:
  color_inputs.append(row([
    widgets[f'focus_channel_{ch}'],
    widgets[f'color_picker_{ch}'],
    widgets[f'color_saturation_{ch}'],
  ]))

right_inputs = column( [widgets['update_image']] + color_inputs, 
  margin=(10,10,10,10),
  height_policy='auto',
  width=100, 
  )
right_inputs.sizing_mode = 'scale_height'

selection_hists = row(
  figures['cluster_hist'], 
  figures['neighbor_hist'],
  sizing_mode='scale_width'
)

all_plots = layout([
  [figures['scatter_plot'], figures['image_plot']], 
  [selection_hists],
  [figures['boxplot'].p]
], sizing_mode='scale_both')


data_layout = layout([ 
  [left_inputs, all_plots, right_inputs],
], sizing_mode='scale_both')


l = layout([[desc], 
            [data_layout]], 
           sizing_mode='scale_both',
)


## -------------------------------------------------------------------
#                    Handle inputs from widgets
## -------------------------------------------------------------------

# Define actions for our buttons: 
# one main function per button that kicks off actions that should take 
# place when a value changes.


def handle_clear_clusters(*args):
  widgets['cluster_select'].active = [] # triggers update scatter


def handle_clear_channels(*args):
  widgets['channels_select'].value = []


def get_selected_clusters():
  cluster_vals = widgets['cluster_select'].active
  # cluster_vals = [f'{d:02d}' for d in cluster_vals]
  cluster_vals = [shared_variables['all_clusters'][d] for d in cluster_vals]
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
      logger.info(f'getting neighbors for cluster {v}')
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
  logger.info(f'total adjacent cells: {n_nbr}')

  update_bbox(shared_variables, figures)
  update_clusters_hist()
  update_neighbors_hist()
  update_scatter()
  figures['scatter_plot'].title.text = "Highlighting %d cells" % n_hl
  figures['cluster_hist'].title.text = "Highlighting %d cells" % n_hl
  figures['neighbor_hist'].title.text = "%d neighboring cells" % n_nbr


def update_scatter():
  logger.info('Updating scatter plot')
  get_selected_clusters()
  n_hl = shared_variables['highlight_cells'].sum()
  n_nbr = shared_variables['neighbor_cells'].sum()

  figures['boxplot'].update_data(shared_variables['adata_data'].loc[shared_variables['highlight_cells']])

  logger.info(f'updating scatter with options: {widgets["cluster_view_opts"].active}')
  logger.info(f'highlighting total {n_hl} cells')

  if 0 in widgets['cluster_view_opts'].active:
    fg_colors = np.array([shared_variables['foreground_color']] * shared_variables['adata_data'].shape[0])
  else:
    fg_colors = np.array(shared_variables['adata_data']['color'])

  fg_colors[~shared_variables['highlight_cells']] = shared_variables['background_color']

  sizes = np.zeros(shared_variables['adata_data'].shape[0])+8
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
    x=shared_variables['adata_data']['coordinates_1'],
    y=shared_variables['adata_data']['coordinates_2']-shared_variables['y_shift'],
    s=sizes,
    index=shared_variables['adata_data']['index_num'],
    annotation=shared_variables['adata_data'][default_annotation], # have to change this to allow switching the shown annotations
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


def update_color(ch):
  """ Attach to a ColorPicker widget """
  def _update_channel_color(attr, old, new):
    color_hex = widgets[f'color_picker_{ch}'].color
    color_rgb = hex_to_dec(color_hex)
    shared_variables['channel_colors'][ch] = np.array(color_rgb)/255.
    logger.info(f'new color for {ch}: {color_hex} {color_rgb}')
  return _update_channel_color


def update_intensity(ch):
  def _update_intensity(attr, old, new):
    print(f'setting channel {ch} saturation to', new)
    shared_variables['saturation_vals'][ch] = new
  return _update_intensity


def handle_set_active_channel(ch):
  def _set_active_channel_internal(v):
    if widgets[f'focus_channel_{ch}'].active:
      widgets[f'focus_channel_{ch}'].button_type = 'danger' # red
      logger.info(f'channel {ch} set to active')
    else:
      widgets[f'focus_channel_{ch}'].button_type = 'success' # green
      logger.info(f'channel {ch} set to inactive')
  return _set_active_channel_internal


def handle_update_image(event):
  update_image_plot(shared_variables, widgets, figure_sources, figures)


def handle_change_input_file(attr, old, new):
  logger.info(f'Setting input file to {widgets["input_file"].value}')
  active_file = widgets["input_file"].value
  set_active_slide(active_file, shared_variables, default_annotation, logger)
  # update_box_selection(None, None, [])
  update_scatter()


# Pin action functions to controls
update_functions['handle_change_input_file'] = handle_change_input_file
update_functions['handle_clear_clusters'] = handle_clear_clusters
update_functions['handle_clear_channels'] = handle_clear_channels
update_functions['update_scatter'] = update_scatter
update_functions['handle_update_image'] = handle_update_image

for ch in shared_variables['all_channels']:
  update_functions[f'update_color_{ch}'] = update_color(ch)
  update_functions[f'update_intensity_{ch}'] = update_intensity(ch)
  update_functions[f'handle_focus_channel_{ch}'] = handle_set_active_channel(ch)
update_functions[f'handle_focus_channel_nuclei'] = handle_set_active_channel('nuclei')

# Tie buttons to functions
widgets['cluster_select'].on_click(lambda event: update_functions['update_scatter']())
widgets['cluster_view_opts'].on_click(lambda event: update_functions['update_scatter']())
widgets['channels_select'].on_change('value', lambda attr, old, new: set_dropdown_menu(shared_variables, widgets))
widgets['clear_clusters'].on_click(update_functions['handle_clear_clusters'])
widgets['clear_channels'].on_click(update_functions['handle_clear_channels'])
widgets['update_image'].on_click(update_functions['handle_update_image'])
widgets['input_file'].on_change('value', update_functions['handle_change_input_file'])

for ch in shared_variables['all_channels']:
  widgets[f'color_picker_{ch}'].on_change('color', update_functions[f'update_color_{ch}'])
  widgets[f'color_saturation_{ch}'].on_change('value', update_functions[f'update_intensity_{ch}'])
  widgets[f'focus_channel_{ch}'].on_click(update_functions[f'handle_focus_channel_{ch}'])
widgets[f'focus_channel_nuclei'].on_click(update_functions[f'handle_focus_channel_nuclei'])

# Only select from foreground "in focus" cells
fg_scatter.data_source.selected.on_change('indices', update_box_selection)

logger.info('Populating initial data')
update_scatter()  # initial load of the data
update_box_selection(None,None,[])

logger.info('adding layout to root')
curdoc().add_root(l)
curdoc().title = "CODEX"
