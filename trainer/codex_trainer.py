from os.path import dirname, join
from bokeh.models.tools import HoverTool, PanTool, ResetTool, WheelZoomTool, BoxSelectTool, PolySelectTool
# from cudf.utils.utils import scalar_broadcast_to

import time

import numpy as np
import scanpy as sc
import pandas as pd
import cv2
# import pandas.io.sql as psql

from bokeh.io import curdoc
from bokeh.layouts import column, row, layout, gridplot
from bokeh.models import (ColumnDataSource, BoxSelectTool, LassoSelectTool, Button, 
                          Div, Select, Slider, TextInput, MultiChoice, MultiSelect, ColorPicker, 
                          Dropdown, Span, CheckboxButtonGroup, Toggle, FileInput, RangeSlider,
                          RadioButtonGroup,
                          CheckboxGroup, Spinner, Panel, Tabs, TapTool)
from bokeh.plotting import figure
from bokeh.models import Range1d

import glob
import seaborn as sns
from matplotlib.colors import rgb2hex
from scipy.sparse import issparse

# from micron2.codexutils import get_images, blend_images
# from micron2.spatial import pull_neighbors
from micron2.codexutils import blend_images, load_nuclei_mask

import pickle
import logging
from .selection_ops_v3 import pull_images, sample_nuclei, gather_features
from .load_data import set_active_slide, load_celltype_config, load_color_config
from datetime import datetime
import os
# from .scatter_gate import ScatterGate
# from .boxplot import BokehBoxPlot

SAMPLE_DIRS = []
SAMPLE_SOURCES = [
  '/common/ingn/CODEX_PROCESSED_DATA/pembroRT_immune/phase1',
  '/common/ingn/CODEX_PROCESSED_DATA/pembroRT_immune/phase2',
  '/common/ingn/CODEX_PROCESSED_DATA/transgenderTMA'
]
# REGISTERED_SAMPLES = pd.read_csv('registered_samples.csv')

for s in SAMPLE_SOURCES:
  SAMPLE_DIRS += glob.glob(f'{s}/*/')
print(SAMPLE_DIRS)

SAMPLE_DIRS = sorted(SAMPLE_DIRS)
def _sample_dir2name(d):
  r = os.path.basename(d[:-1])
  print(f'getting display name for {d}: {r}')

  return r

def filter_dirs(d):
  has_centroid_file = len(glob.glob(f'{d}/*centroids.csv')) > 0
  return has_centroid_file

SAMPLE_DIRS = {_sample_dir2name(s):s for s in SAMPLE_DIRS if filter_dirs(s)} 


print(SAMPLE_DIRS)


# https://docs.bokeh.org/en/latest/docs/gallery/color_sliders.html
def hex_to_dec(hex):
  red = ''.join(hex.strip('#')[0:2])
  green = ''.join(hex.strip('#')[2:4])
  blue = ''.join(hex.strip('#')[4:6])
  return (int(red, 16), int(green, 16), int(blue,16), 255)


""" 
Make a little class for each 'pane' that handles:
  1. set up , empty.
  2. updates
  3. resetting. ie. revert to a default nascent state , populated with data.
  4. providing hooks to the widgets/figures/etc. for adding to the layout
""" 
TOOLTIPS=[
    # ("Annotation", "@annotation"),
    # ("Training", "@training"),

    ## We want to add columns dynamically, not have dummy columns with 0's at first.
    ("Annotation", "@annotation"),
    ("Prediction", "@prediction"),
    # ("celltype", "@celltype"),
    # ("subtype", "@subtype"),
    # ("niche_labels", "@niche_labels"),
    # ("state_probs", "@state_probs"),
    ("Index", "@index"),
    ("x", "@x"),
    ("y", "@y"),
]
class ScatterImagePane:
  """ The main attraction
  """
  def __init__(self, logger):
    self.logger = logger
    self.width = 300
    self.height = 300
    # These are out here because we have two glyphs on the plot and only want tooltips for one.
    self.hover_tool = HoverTool(tooltips=TOOLTIPS)
    self.tools = [self.hover_tool, PanTool(), ResetTool(), 
                  WheelZoomTool(), BoxSelectTool(), TapTool()]
    self.p = figure(plot_height=self.height, plot_width=self.width, 
                    title="", toolbar_location='left', 
                    x_range=Range1d(), y_range=Range1d(),
                    tools = self.tools, 
                    sizing_mode="scale_both",
                    match_aspect=True,
                    output_backend='webgl')
    self.p.select(BoxSelectTool).select_every_mousemove = False
    self.p.select(LassoSelectTool).select_every_mousemove = False

    self.scatter_source = ColumnDataSource(
      data=dict(
          # x, y, s, index, and color are always here
          x=[], y=[], s=[], index=[], color=[], alpha=[],
          ## these need to get added in later, at the users discretion
          annotation=[], training=[], predict_prob=[], 
        )
      )

    self.image_source = ColumnDataSource(
      data=dict(
          # value holds pixel values
          value= [], 
          # dw, dh controls aspect ratio
          dw= [], dh= [], 
          # x0, y0 are offsets from 0, to produce images aligned with the scatter points
          x0= [], y0= []
        )
      )
    
    self.region_source = ColumnDataSource(
      data=dict(
        xs=[],
        ys=[],
        alpha=[],
        color=[]
      )

    )

    self.scatter = self.p.scatter(x="x", y="y", source=self.scatter_source, 
        radius="s", color="color", line_color=None, 
        fill_alpha="alpha", 
        )
    self.hover_tool.renderers = [self.scatter]
    self.img_plot = self.p.image_rgba(image='value', source=self.image_source,
                    x='x0',y='y0',dw='dw',dh='dh')
    self.img_plot.level = 'underlay'

    print(dir(self.scatter.nonselection_glyph))
    self.scatter.nonselection_glyph.fill_alpha = 1.

    # self.regions = self.p.multi_polygons(xs="xs", ys="ys", alpha="alpha", 
    #     line_color="color", 
    #     fill_color=None,
    #     line_width=4,
    #     source=self.region_source)
    # self.img_plot.level = 'overlay'

  def reset(self):
    pass


CELLTYPE_DEFS = {
'pembroRT TLS': '/home/ingn/devel/micron2/trainer/configs/pembro_TLS_panel_celltypes.json',
'transgender TMA': '/home/ingn/devel/micron2/trainer/configs/transgenderTMA_celltypes.json',
}

COLOR_CONFIGS = {
'pembroRT TLS': '/home/ingn/devel/micron2/trainer/configs/pembro_TLS_panel_colors.csv',
'transgender TMA': '/home/ingn/devel/micron2/trainer/configs/transgenderTMA_colors.csv',
}

CLUSTER_VIEW_OPTS = [
  'Hide scatter plot'
]
class ScatterSettings:
  def __init__(self, logger):
    self.logger = logger

    self.sample_select = Select(options=list(SAMPLE_DIRS.keys()), 
                                # active=list(SAMPLE_DIRS.keys())[0], 
                                css_classes=["my-widgets"])

    # Setter for active color annotation
    self.cluster_select = CheckboxGroup(active=[], labels=[], 
                                        margin=(10,10,10,10),
                                        height=300,
                                        css_classes=["my-widgets"])

    self.cluster_view_opts = CheckboxGroup(labels=CLUSTER_VIEW_OPTS, 
                                           active=[],
                                           margin=(10,10,10,10),
                                           css_classes=["my-widgets"])

    # self.input_file = TextInput(placeholder='sample directory')

    # self.celltype_config_file = TextInput(placeholder='celltype config',
    #   value='/home/ingn/devel/micron2/trainer/configs/pembro_TLS_panel_celltypes.json')
    # self.channel_config_file = TextInput(placeholder='channel config',
    #   value='/home/ingn/devel/micron2/trainer/configs/pembro_TLS_panel_colors.csv')

    self.celltype_config_file = Select(options=list(CELLTYPE_DEFS.keys()), 
                                css_classes=["my-widgets"])

    self.channel_config_file = Select(options=list(COLOR_CONFIGS.keys()), 
                                css_classes=["my-widgets"])

    self.load_data_button = Button(label='Initialize', button_type='success',
                               margin=(10,10,10,10),)

    self.next_image = Button(label='Next image', button_type='success',
                               margin=(10,10,10,10),)

    self.dot_size = Spinner(low=1, high=10, step=1, value=3, 
                            css_classes=["my-widgets"], name='Dot size', 
                            width=100)
    self.choose_annotation = Select(title="Choose annotation:", 
                               options=[], css_classes=['my-widgets'])

    self.commit_annotations = Button(label='Commit annotations', button_type='success',
                               margin=(10,10,10,10),)
    self.fit_predict = Button(label='Fit & Predict', button_type='success',
                               margin=(10,10,10,10),)
    self.sample_rate = Spinner(low=0.01, high=1, step=0.01, value=0.05, 
                            css_classes=["my-widgets"], name='sample rate', 
                            width=100)

    self.active_regions = []


  @property
  def layout(self):
    l = column([
        self.sample_select,
        # self.input_file,
        self.celltype_config_file,
        self.channel_config_file,
        self.load_data_button,
        self.next_image,
        self.cluster_view_opts,
        self.dot_size,
        self.choose_annotation,
        self.cluster_select, 
        self.commit_annotations,
        self.sample_rate,
        self.fit_predict,
        # self.choose_region, 
        # self.region_select
      ],
      margin=(10,10,10,10),
      width=150)
    return l

  def reset(self):
    pass




class ImageColorSettings:
  def __init__(self, logger):
    self.logger = logger
    self.channels = []

    self.view_selector = RadioButtonGroup(labels=['Overview', '1x'], active=0)

    self.celltype_selector = Select(title="Annotate celltype:", 
                                    options=[], css_classes=['my-widgets'])

    self.update_image = Button(label='Update drawing', button_type='success',
                               margin=(10,10,10,10),)
    
    self._focus_channel_buttons = dict(
      nuclei = Toggle(label='nuclei', button_type='success', width=75)
    )
    # Register a function to deal with inputs from this button
    self.focus_channel_buttons('nuclei').on_click(self.handle_toggle_focus_factory('nuclei'))

    # These don't get handlers
    self._channel_color_pickers = dict(
      nuclei = ColorPicker(width=50, color='#ad9703', css_classes=["my-widgets"])
    )

    self._channel_low_thresholds = dict(
      # nuclei = Spinner(low=0, high=int(2**16), step=5, #tags=[ch], 
      #                  width=75, value=0, css_classes=["my-widgets"])
    )
    self._channel_high_thresholds = dict(
      # nuclei = Spinner(low=0, high=int(2**16), step=5, #tags=[ch], 
      #                  width=75, value=0, css_classes=["my-widgets"])
    )

    self.resize = Spinner(low=0.1, high=1, step=0.05, value=0.2, 
                            css_classes=["my-widgets"], name='Dot size', 
                            width=100)
    self.layout = self.default_color_widgets
    self.active_channels = dict(nuclei=False)


  def handle_toggle_focus_factory(self, ch):
    # Each channel focuser needs its own little function
    # that's tied specifically to that button.
    self.logger.info(f'Creating function handler for focus button {ch}')
    def _toggle_focus(attr):
      button = self.focus_channel_buttons(ch)
      if button.active:
        self.logger.info(f'Setting channel {ch} to active')
        button.button_type = 'danger' # red
        self.active_channels[ch] = True
      else:
        self.logger.info(f'Setting channel {ch} to inactive')
        button.button_type = 'success' # green
        self.active_channels[ch] = False
    return _toggle_focus


  def focus_channel_buttons(self, channel):
    return self._focus_channel_buttons[channel]

  def channel_color_pickers(self, channel):
    return self._channel_color_pickers[channel]

  def channel_low_thresholds(self, channel):
    return self._channel_low_thresholds[channel]

  def channel_high_thresholds(self, channel):
    return self._channel_high_thresholds[channel]


  def add_channel(self, ch, default_color, default_low=0, default_high=0):
    if ch not in self.channels:
      self.logger.info(f'Adding color widgets for channel {ch}')
      self._focus_channel_buttons[ch] = Toggle(label=ch, button_type='success',
         width=75)
      self._channel_color_pickers[ch] = ColorPicker(width=50, color=default_color, 
        css_classes=["my-widgets"])
      self._channel_low_thresholds[ch] = Spinner(low=0, high=int(2**16), step=5,  
        width=75, value=default_low, css_classes=["my-widgets"])
      self._channel_high_thresholds[ch] = Spinner(low=0, high=int(2**16), step=5, 
        width=75, value=default_high, css_classes=["my-widgets"])
      self.focus_channel_buttons(ch).on_click(self.handle_toggle_focus_factory(ch))
      self.channels.append(ch)
      self.active_channels[ch] = False


  def get_color_widget(self, ch):
    return row([self.focus_channel_buttons(ch),
                self.channel_color_pickers(ch),
                self.channel_low_thresholds(ch),
                self.channel_high_thresholds(ch)])


  def default_color_widget_list(self):
    nuclei_settings = row([
      self.focus_channel_buttons('nuclei'),
      self.channel_color_pickers('nuclei'),
    ])
    return [self.view_selector, self.celltype_selector, self.update_image, self.resize, nuclei_settings]


  @property
  def default_color_widgets(self):
    l = column(self.default_color_widget_list())
    return l


  def get_image_color_widgets(self):
    widgets = []
    for ch in self.channels:
      widgets.append(self.get_channel_color_widget(ch))
    return widgets


  def reset(self):
    pass





class CodexTrainer:

  def __init__(self):
    self.shared_var = {}
    self.figure_sources = {}
    self.current_images = {}
    self.active_bbox = None

    self.make_logger()
    self.logger.info('Welcome')

    self.scatter_widgets = ScatterSettings(self.logger)
    self.image_colors = ImageColorSettings(self.logger)
    self.scatter_image = ScatterImagePane(self.logger)

    self.register_callbacks()

    # self.set_input_file(None, None, '/storage/codex/preprocessed_data/pembro_TLS_panel/210702_PembroRT_Cas19_TLSpanel_reg2')
    # self.set_channel_config(None, None, '/home/ingn/devel/micron2/trainer/configs/pembro_TLS_panel_colors.csv')
    # self.set_celltype_config(None, None, '/home/ingn/devel/micron2/trainer/configs/pembro_TLS_panel_celltypes.json')


  
  def make_logger(self):
    self.logger = logging.getLogger('vis')
    self.logger.setLevel('INFO')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # self.logger.addHandler(sh)


  def register_callbacks(self):
    self.scatter_widgets.load_data_button.on_click(lambda a: self._handle_load_data())

    # self.scatter_widgets.cluster_view_opts.on_click(lambda x: self.update_scatter())
    # self.scatter_widgets.choose_annotation.on_change('value', self.set_annotation_groups)
    # self.scatter_widgets.dot_size.on_change('value', lambda a,o,n: self.update_scatter())

    # self.scatter_widgets.cluster_select.on_click(lambda a: self.update_scatter())
    # self.scatter_widgets.region_select.on_click(lambda a: self.draw_region())

    self.scatter_widgets.next_image.on_click(lambda a: self.cycle_image())
    self.scatter_widgets.commit_annotations.on_click(lambda a: self._commit_annotations_v2())
    self.scatter_widgets.fit_predict.on_click(lambda a: self._fit_predict())

    # self.scatter_image.scatter.data_source.selected.on_change('indices', self.update_box_selection)

    # This should be the main function for annotating a cell
    self.scatter_image.scatter.data_source.selected.on_change('indices', self.stage_annotation)

    self.image_colors.update_image.on_click(self.update_image_plot)
    self.image_colors.celltype_selector.on_change('value', self.set_active_annotation)


  def _handle_load_data(self):
    # we have to set these files in this order:
    #input_dir = self.scatter_widgets.input_file.value

    self.logger.info('Pulling input dir from dropdown selector')
    input_dir = SAMPLE_DIRS[self.scatter_widgets.sample_select.value]

    self.logger.info(f'Reading input dir: {input_dir}')
    self.set_input_file(None, None, input_dir)

    channel_file = COLOR_CONFIGS[self.scatter_widgets.channel_config_file.value]
    self.logger.info(f'Channel color config: {channel_file}')
    self.set_channel_config(None, None, channel_file)

    celltype_file = CELLTYPE_DEFS[self.scatter_widgets.celltype_config_file.value]
    self.logger.info(f'Celltype config: {celltype_file}')
    self.set_celltype_config(None, None, celltype_file)



  # def _commit_annotations(self):
  #   self.logger.info('Committing annotated cells')
  #   self.logger.info(f'sampling rate: {self.scatter_widgets.sample_rate.value}')
  #   self.logger.info(f'sampling from {self.shared_var["coords"].shape[0]} cells')
  #   # selection_ops_v3.py
  #   channel_stacks, annot = sample_nuclei(self.shared_var['image_sources'], 
  #                                         self.shared_var['coords'],
  #                                         np.array(self.shared_var['cell_data']['annotation']),
  #                                         self.logger,
  #                                         sample_rate = self.scatter_widgets.sample_rate.value
  #   )

  #   channel_order = np.array(self.shared_var['all_channels'])
  #   for ch in channel_order:
  #     slow = self.image_colors.channel_low_thresholds(ch).value
  #     shigh = self.image_colors.channel_high_thresholds(ch).value
  #     self.logger.info(f'saturating channel {ch}: {slow} - {shigh}')
  #     stacked_imgs = channel_stacks[ch]
  #     # stacked_imgs = stacked_imgs - slow
  #     stacked_imgs[stacked_imgs < slow] = 0
  #     stacked_imgs[stacked_imgs > shigh] = shigh
  #     stacked_imgs = stacked_imgs / shigh
  #     channel_stacks[ch] = (stacked_imgs * 255).astype(np.uint8)

  #   training_images = np.stack([channel_stacks[ch] for ch in channel_order], axis=-1)
  #   annot = np.array(annot)

  #   now = f'collection-{datetime.now().strftime("%y-%h-%d-%H-%M-%S")}'
  #   annotation_base_dir = f'{self.shared_var["data_dir"]}/{now}'
  #   if not os.path.exists(annotation_base_dir):
  #     os.makedirs(annotation_base_dir)

  #   annotation_base = f'{annotation_base_dir}/training_cells'
  #   self.logger.info(f'Writing annotated cell images to {annotation_base}_*')
  #   self.logger.info(f'images: {training_images.shape}')
  #   self.logger.info(f'annotations: {annot.shape}')

  #   np.save(f'{annotation_base}_images.npy', training_images)
  #   np.save(f'{annotation_base}_annots.npy', annot)
  #   np.save(f'{annotation_base}_channels.npy', channel_order)

  def _fit_predict(self):
    self.logger.info('Fitting classifier & running prediction on the current image')

    cell_data = self.shared_var['cell_data'].copy()

    # There's some bullshit happening here , if we use:
    #    annotated_classes = cell_data['annotation'].dropna().unique()
    # there's a blank class returned for some reason
    annotated_classes = []
    for u in cell_data['annotation'].dropna().unique():
      if u=='':continue
      c = np.sum(cell_data['annotation'] == u)
      self.logger.info(f'Annotation for: {u}: {c}')
      annotated_classes.append(u)

    feature_df = pd.DataFrame(index=cell_data.index, 
                            columns=self.shared_var['available_channels'])
    current_tile_cells = self.shared_var['visible_cells']
    annotated_cells = cell_data.loc[cell_data['annotation'].isin(annotated_classes)].index
    self.logger.info(f'visible cells: {len(current_tile_cells)}')
    self.logger.info(f'all annotated cells: {len(annotated_cells)}')
    use_cells = list(current_tile_cells) + list(annotated_cells)

    feature_df = feature_df.loc[feature_df.index.isin(use_cells)]
    feature_df['annotation'] = cell_data.loc[feature_df.index, 'annotation']

    self.logger.info(f'Set up naive feature_df dict: {feature_df.shape}') 
    


  def _commit_annotations_v2(self):
    self.logger.info('Committing annotated cells') 
    # annotation = np.array(self.shared_var['cell_data']['annotation']),

    now = f'annotation-{datetime.now().strftime("%y-%h-%d-%H-%M-%S")}'
    annotation_file = f'{self.shared_var["data_dir"]}/{now}.csv'
    self.logger.info(f'Writing cell data to {annotation_file}')
    self.shared_var['cell_data'].to_csv(annotation_file)




  def stage_annotation(self, attr, old, new):
    if len(new) == 0:
      self.logger.info(f'nothing selected.')
      return 0

    active_annotation_class = self.image_colors.celltype_selector.value
    self.logger.info(f'annotating for class: {active_annotation_class}')
    self.logger.info(f'{attr}, {old}, {new}')

    active_annotation_color = self.shared_var['celltype_colors'][active_annotation_class]

    new = new[0]
    selected_cell_id = self.scatter_image.scatter_source.data['index'][new]
    self.logger.info(f'cell index: {selected_cell_id}')

    # here is where we record the annotations in terms of the global coordinates
    self.shared_var['cell_data'].loc[selected_cell_id, 'annotation'] = active_annotation_class

    # update the color of the cell we just selected
    data = self.scatter_image.scatter_source.data.copy()
    self.logger.info(f'setting color: {active_annotation_color}')
    data['color'][new] = hex_to_dec(active_annotation_color)[:3]
    self.scatter_image.scatter_source.data = data


  def set_active_annotation(self, attr, old, new):
    self.logger.info(f'Setting annotation to highlight celltype: {new}')
    active_channels = self.shared_var['celltype_channels'][new]
    self.logger.info(f'corresponding channels: {active_channels}')

    # De-select all channels first
    for ch in self.shared_var['all_channels']:
      button = self.image_colors.focus_channel_buttons(ch)
      self.logger.info(f'setting channel {ch} to inactive')
      self.image_colors.active_channels[ch] = False
      button.button_type = 'success' # green
      picker = self.image_colors.channel_color_pickers(ch)
      picker.color = hex_to_dec(self.shared_var['default_channel_color'])[:3]

    # Set the active channels and change the color
    for i,ch in enumerate(active_channels):
      self.logger.info(f'setting active channel: {i} {ch}')
      self.image_colors.active_channels[ch] = True
      button = self.image_colors.focus_channel_buttons(ch)
      button.button_type = 'danger' # red
      picker = self.image_colors.channel_color_pickers(ch)
      picker.color = hex_to_dec(self.shared_var['color_wheel'][i])[:3]



  def set_celltype_config(self, attr, old, new):
    load_celltype_config(new, self.shared_var, self.logger)
    self.image_colors.celltype_selector.options = self.shared_var['celltype_choices']
    self.image_colors.celltype_selector.value = self.shared_var['celltype_choices'][0]
    self.set_active_annotation(None, None, self.shared_var['celltype_choices'][0])


  def set_channel_config(self, attr, old, new):
    load_color_config(new, self.shared_var, self.logger)
    available_images = [ch for ch, val in self.shared_var['image_sources'].items() if val is not None]
    self.shared_var['available_channels'] = available_images

    self.logger.info(f'populating available channels...')
    color_widgets = self.image_colors.default_color_widget_list()
    for ch in available_images:
      default_color = self.shared_var['channel_colors'][ch]
      default_low, default_high = self.shared_var['saturation_vals'][ch]
      self.image_colors.add_channel(ch, default_color, default_low, default_high)
      color_widgets.append(self.image_colors.get_color_widget(ch))
    self.image_colors.layout.children = color_widgets


  def set_input_file(self, attr, old, new):
    self.active_file = new
    self.logger.info(f'Setting input file to: {new}')
    set_active_slide(self.active_file, self.shared_var, self.logger)


  def reset_to_overview(self):
    """
    0. stash settings to reset current view. keep the image data around. 
    1. set display height and width to match the overall image
    2. set image source with saved overall image data, blended into the currently active color set
    3. set coordinates shown to all, with the currently active dot annotation
    4. view_toggle_botton: set to "overall"
    """

    pass


  def reset_to_zoomedview(self):

    pass



  def cycle_image(self):
    overview_mode = self.image_colors.view_selector.active==0
    self.logger.info(f'upading image in overview mode: {overview_mode}')

    if overview_mode:
      bbox = None
    else:
      bbox = next(self.shared_var['bbox_generator'])
      self.active_bbox = bbox

    self.logger.info(f'Cycling image. next bbox: {bbox}')

    rows = np.array(self.shared_var['coords'][:,0])
    cols = np.array(self.shared_var['coords'][:,1])

    self.current_images = pull_images(self.shared_var['image_sources'], bbox, self.logger, 
                                      resize = self.image_colors.resize.value if overview_mode else 1)

    for i, (k,v) in enumerate(self.current_images.items()):
      self.logger.info(f'{i} image {k}: {v.shape}')

    if overview_mode:
      # current_shape = list(self.current_images.keys())[0]
      current_shape = self.current_images[list(self.current_images.keys())[0]].shape
      self.logger.info(f'Setting bbox for overview based on loaded image: {current_shape}')
      self.active_bbox = [0, current_shape[0], 0, current_shape[1]]
      bbox = self.active_bbox
      self.logger.info(f'active bbox: {self.active_bbox}, bbox: {bbox}')

    visible_cells_row = (rows > bbox[0]) & (rows < bbox[1])
    visible_cells_col = (cols > bbox[2]) & (cols < bbox[3])
    visible_cells = visible_cells_col & visible_cells_row

    self.shared_var['visible_cells'] = self.shared_var['cell_names'][visible_cells]

    self.logger.info(f'region contains {np.sum(visible_cells)} cells')

    x = cols[visible_cells]
    y = rows[visible_cells]

    coord_shift_x = bbox[2]
    coord_shift_y = bbox[0]

    data = dict(
      # x = x - coord_shift_x,
      # y = y - coord_shift_y,
      x = rows[visible_cells] - bbox[0],
      y = cols[visible_cells] - bbox[2],
      s = np.zeros_like(x, dtype=np.uint8)+self.scatter_widgets.dot_size.value,
      alpha = np.ones_like(x, dtype=np.uint8),
      color = np.array([hex_to_dec(self.shared_var['default_dot_color'])[:3]] * np.sum(visible_cells)),
      index = self.shared_var['cell_names'][visible_cells]
    )
  # update the axis ranges to contain the whole bbox.

    self.scatter_image.p.x_range.start = bbox[2] - coord_shift_x
    self.scatter_image.p.x_range.end = bbox[3] - coord_shift_x
    self.scatter_image.p.y_range.start = bbox[0] - coord_shift_y
    self.scatter_image.p.y_range.end = bbox[1] - coord_shift_y
    self.scatter_image.scatter_source.data = data

    self.update_image_plot()




  # def update_scatter(self):
  #   """ Parse all the applicable requests and update the scatter figure
  #   """
  #   # Do the data operations, then dispatch to a method in the ImageScatterPane class
  #   # to handle updating the data source and redrawing
  #   self.logger.info('Updating scatter plot')
  #   self.get_selected_clusters()
  #   n_hl = self.shared_var['highlight_cells'].sum()

  #   colors = np.array(self.shared_var['color'], dtype=object)
  #   colors[~self.shared_var['cluster_selection']] = self.shared_var['background_color']
  #   sizes = np.zeros_like(colors, dtype=np.uint8)+self.scatter_widgets.dot_size.value
  #   alpha = np.ones_like(sizes)
  #   alpha[~self.shared_var['highlight_cells']] = 0

  #   self.scatter_image.p.title.text = f"Highlighting {n_hl} cells"
  #   data =dict(
  #     x=self.shared_var['adata_data']['coordinates_1'],
  #     y=self.shared_var['adata_data']['coordinates_2']-self.shared_var['y_shift'],
  #     s=sizes,
  #     alpha=alpha,
  #     index=self.shared_var['adata_data']['index_num'],
  #     color=colors,
  #   )
  #   ## Again, here we want to add these dynamically when requested, not using dummy values to pad missing
  #   for col in ['niche_labels', 'celltype', 'subtype', 'state_probs']:
  #     if col in self.shared_var['adata_data'].columns:
  #       self.logger.info(f'Setting values for hover tool field: {col}')
  #       data[col] = self.shared_var['adata_data'][col]
  #     else:
  #       self.logger.info(f'Using FILLER values for hover tool field: {col}')
  #       data[col] = [0]*self.shared_var['n_cells']

  #   self.scatter_image.scatter_source.data = data
  #   if self.reframe:
  #     update_bbox(self.shared_var, self.scatter_image.p, self.logger)
  #   self.reframe=False

  #   # Allow hiding points
  #   if 0 in self.scatter_widgets.cluster_view_opts.active:
  #     self.logger.info('Hiding scatter plot')
  #     self.scatter_image.img_plot.level = 'overlay'
  #   else:
  #     self.logger.info('Showing scatter plot')
  #     self.scatter_image.img_plot.level = 'underlay'


  # def update_box_selection(self, attr, old, new):
  #   self.logger.info('Handling bbox selection')
  #   box_selection = np.ones(self.shared_var['n_cells'], dtype=bool)
  #   if len(new) > 0:
  #     box_selection = np.zeros(self.shared_var['n_cells'], dtype=bool)
  #     box_selection[new] = 1

  #   self.shared_var['box_selection'] = box_selection
  #   self.logger.info(f'bbox selection: {box_selection.sum()}')
  #   self.shared_var['highlight_cells'] = box_selection & self.shared_var['cluster_selection']
  #   self.reframe=True
  #   self.update_scatter()


  def get_selected_clusters(self):
    selected_clusters = self.scatter_widgets.cluster_select.active
    selected_clusters = [self.scatter_widgets.u_groups[i] for i in selected_clusters]
    self.logger.info(f'Setting selected clusters to {selected_clusters}')
    cluster_selection = np.zeros(self.shared_var['n_cells'], dtype=bool)
    if len(selected_clusters) == 0:
      self.logger.info('No clusters selected for focus')
      cluster_selection[:] = 1
    else:
      for v in selected_clusters:
        cluster_idx = self.shared_var['clusters']==v
        self.logger.info(f'highlighting {np.sum(cluster_idx)} cells from cluster {v}')
        cluster_selection[cluster_idx] = 1
      # TODO @nathanin take care of neighbors here

    self.shared_var['cluster_selection'] = cluster_selection
    self.shared_var['highlight_cells'] = self.shared_var['cluster_selection']
    

  def update_image_plot(self):
    self.logger.info('Updating image plot')
    
    use_channels = [ch for ch in self.shared_var['all_channels'] if self.image_colors.active_channels[ch]]
    images = np.dstack([self.current_images[ch] for ch in use_channels])

    self.logger.info(f'Blending images: {images.shape}')

    colors = []
    for ch in use_channels:
      c = self.image_colors.channel_color_pickers(ch).color
      chc = np.array(hex_to_dec(c))
      self.logger.info(f'drawing channel {ch} with color {chc}')
      colors.append(chc)
    colors = np.stack(colors, axis=0)

    saturation = []
    for i,ch in enumerate(use_channels):
      # slow, shigh = self.shared_var['saturation_vals'][ch] 
      slow = self.image_colors.channel_low_thresholds(ch).value
      shigh = self.image_colors.channel_high_thresholds(ch).value
      if (shigh is None) or (shigh == 0):
        img = images[:,:,i]
        if img.sum() == 0:
          shigh = 0
        else:
          vals = img.ravel()[img.ravel()>0]
          shigh = np.quantile(vals, 0.999)

      if (slow is None) or (slow == 0):
        img = images[:,:,i]
        if img.sum() == 0:
          slow = 0
        else:
          slow = int(shigh/256)

      # make sure the saturation val widget reflects the value being drawn
      self.image_colors.channel_low_thresholds(ch).value = slow
      self.image_colors.channel_high_thresholds(ch).value = shigh
      self.logger.info(f'channel {ch} saturating at [{slow}, {shigh}]')
      self.shared_var['saturation_vals'][ch] = (slow, shigh)
      saturation.append((slow,shigh))

    if self.image_colors.focus_channel_buttons('nuclei').active:
      nuclei_path = self.shared_var['nuclei_path']
      self.logger.info(f'Loading nuclei from {nuclei_path}')
      nuclei = load_nuclei_mask(nuclei_path, self.shared_var['bbox'])
      nuclei_color = self.image_colors.channel_color_pickers('nuclei').color
      nuclei_color = hex_to_dec(nuclei_color)
      self.logger.info(f'drawing nuclei with color {nuclei_color}')
    else:
      nuclei = None
      nuclei_color = None
    blended = blend_images(images, saturation_vals=saturation, colors=colors, 
                           nuclei=nuclei, nuclei_color=nuclei_color)
    blended = blended[::-1,:] # flip
    self.logger.info(f'created blended image: {blended.shape}')

    ## Set the aspect ratio according to the selected area
    bbox = self.active_bbox
    dw = bbox[3] - bbox[2]
    dh = bbox[1] - bbox[0]
    self.logger.info(f'bbox would suggest the image shape to be: {dw} x {dh}')

    self.logger.info(f'update image aspect ratio: {dw} / {dh}')
    self.scatter_image.image_source.data = {
      'value': [blended],
      ## Aspect ratio preserved, stretching to match scatter coordinates
      'dw': [dw], 
      'dh': [dh],
      ## Offset to match the scatter plotted dots
      # 'x0': [bbox_plot[0]],
      # 'y0': [bbox_plot[2]],
      # 'x0': [bbox[0]],
      # 'y0': [bbox[2]],
      'x0': [0],
      'y0': [0],
      }
    self.scatter_image.img_plot.level = 'underlay'


  @property
  def data_layout(self):
    l = layout([
      [
        self.scatter_widgets.layout,
        self.scatter_image.p,
        self.image_colors.layout
      ]
    ])
    return l
