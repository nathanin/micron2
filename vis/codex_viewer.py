from os.path import dirname, join
from bokeh.models.tools import HoverTool, PanTool, ResetTool, WheelZoomTool, BoxSelectTool, PolySelectTool
from cudf.utils.utils import scalar_broadcast_to

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
                          CheckboxGroup, Spinner, Panel, Tabs, )
from bokeh.plotting import figure
from bokeh.models import Range1d

import glob
import seaborn as sns
from matplotlib.colors import rgb2hex
from scipy.sparse import issparse

# from micron2.codexutils import get_images, blend_images
# from micron2.spatial import pull_neighbors
from micron2.codexutils import blend_images, load_nuclei_mask

import logging
from .selection_ops_v2 import update_bbox, maybe_pull
from .load_data import set_active_slide
# from .scatter_gate import ScatterGate
# from .boxplot import BokehBoxPlot


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
    # ("Predict prob", "@predict_prob"),
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
    self.hover_tool = HoverTool(tooltips=TOOLTIPS)
    self.tools = [self.hover_tool, PanTool(), ResetTool(), 
                  WheelZoomTool(), BoxSelectTool()]
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
          x=[], y=[], s=[], index=[], color=[]
          ## these need to get added in later, at the users discretion
          # annotation=[], training=[], predict_prob=[], 
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

    self.scatter = self.p.scatter(x="x", y="y", source=self.scatter_source, 
                          radius='s', color="color", line_color=None)
    self.hover_tool.renderers = [self.scatter]
    self.img_plot = self.p.image_rgba(image='value', source=self.image_source,
                    x='x0',y='y0',dw='dw',dh='dh')
    self.img_plot.level = 'underlay'

  def reset(self):
    pass





CLUSTER_VIEW_OPTS = [
  'Join focused clusters',
  'Show neighbors',
  'Hide scatter plot'
]
class ScatterSettings:
  def __init__(self, logger):
    self.logger = logger
    self.cluster_select = CheckboxGroup(active=[], labels=[], 
                                        margin=(10,10,10,10),
                                        css_classes=["my-widgets"])
    self.clear_clusters = Button(label='Clear focused cells', 
                                 button_type='success')
    self.cluster_view_opts = CheckboxGroup(labels=CLUSTER_VIEW_OPTS, 
                                           active=[],
                                           margin=(10,10,10,10),
                                           css_classes=["my-widgets"])
    self.input_file = TextInput(placeholder='Enter full path to sample h5ad')
    self.dot_size = Spinner(low=1, high=30, step=2, value=10, 
                            css_classes=["my-widgets"], name='Dot size', 
                            width=100)
    self.choose_annotation = Select(title="Choose annotation:", 
                               options=[], css_classes=['my-widgets'])
    self.hover_tooltips = MultiChoice(value=[], options=[])

  @property
  def layout(self):
    l = column([
        self.input_file,
        self.cluster_view_opts,
        self.dot_size,
        self.hover_tooltips,
        self.clear_clusters,
        self.choose_annotation,
        self.cluster_select, 
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
    self.update_image = Button(label='Update drawing', button_type='success',
                               margin=(10,10,10,10),)
    
    self._focus_channel_buttons = dict(
      nuclei = Toggle(label='nuclei', button_type='success', width=75)
    )
    # Register a function to deal with inputs from this button
    self.focus_channel_buttons('nuclei').on_click(self.handle_toggle_focus_factory('nuclei'))

    # These don't get handlers
    self._channel_color_pickers = dict(
      nuclei = ColorPicker(width=50, color='#f2e70a', css_classes=["my-widgets"])
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


  def add_channel(self, ch, default_color):
    if ch not in self.channels:
      self.logger.info(f'Adding color widgets for channel {ch}')
      self._focus_channel_buttons[ch] = Toggle(label=ch, button_type='success',
         width=75)
      self._channel_color_pickers[ch] = ColorPicker(width=50, color=default_color, 
        css_classes=["my-widgets"])
      self._channel_low_thresholds[ch] = Spinner(low=0, high=int(2**16), step=5,  
        width=75, value=0, css_classes=["my-widgets"])
      self._channel_high_thresholds[ch] = Spinner(low=0, high=int(2**16), step=5, 
        width=75, value=0, css_classes=["my-widgets"])
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
    return [self.update_image, self.resize, nuclei_settings]


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








class CodexViewer:
  """ Container that holds data and settings needed by all widgets/figure panes

  Responsibilities:
    1. set up, initial loading and constructing of all figure panes, hooking widgets to functions
    1a. handle parsing and loading data, and sticking it into the right places
    2. dispatch work from widget/interaction events
  """ 
  def __init__(self):
    self.shared_var = {}
    self.figure_sources = {}

    self.make_logger()
    self.logger.info('Welcome')


    self.scatter_widgets = ScatterSettings(self.logger)
    self.image_colors = ImageColorSettings(self.logger)
    self.scatter_image = ScatterImagePane(self.logger)

    self.register_callbacks()

  
  def make_logger(self):
    self.logger = logging.getLogger('vis')
    self.logger.setLevel('INFO')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # self.logger.addHandler(sh)


  def register_callbacks(self):
    self.scatter_widgets.input_file.on_change('value', self.set_input_file)
    self.scatter_widgets.choose_annotation.on_change('value', self.set_annotation_groups)
    self.scatter_widgets.dot_size.on_change('value', lambda a,o,n: self.update_scatter())
    self.scatter_widgets.cluster_select.on_click(lambda a: self.update_scatter())
    self.scatter_image.scatter.data_source.selected.on_change('indices', self.update_box_selection)
    self.image_colors.update_image.on_click(self.update_image_plot)
    self.scatter_widgets.hover_tooltips.on_change('value', self.add_hover_data)


  def add_hover_data(self, val):
    hover_cols = self.scatter_widgets.hover_tooltips.value
    tooltips = [
      ("Index", "@index"),
      ("x", "@x"),
      ("y", "@y"),
    ]
    for col in hover_cols:
      tooltips.insert(0, (col, f'@{col}'))
      if col not in self.scatter_image.scatter_source.data.keys():
        self.scatter_image.scatter_source.data[col] = self.shared_var['adata_data'][col].tolist()


  def set_annotation_groups(self, attr, old, new):
    self.logger.info(f'Setting active annotation: {new}')
    annotation = self.shared_var['adata_data'][new]
    self.scatter_widgets.u_groups = list(np.unique(annotation))
    self.scatter_widgets.cluster_select.labels = self.scatter_widgets.u_groups
    self.shared_var['clusters'] = annotation

    n_clusters = len(self.scatter_widgets.u_groups)
    if f'{new}_colors' in self.shared_var['adata'].uns.keys():
      cluster_colors = self.shared_var['adata'].uns[f'{new}_colors']
      color_map = {k: v for k, v in zip(self.scatter_widgets.u_groups, cluster_colors)}
    else:
      cluster_colors = sns.color_palette('Set1', n_clusters)
      cluster_colors = np.concatenate([cluster_colors, np.ones((n_clusters, 1))], axis=1)
      color_map = {k: rgb2hex(v) for k, v in zip(self.scatter_widgets.u_groups, cluster_colors)}

    self.shared_var['color'] = [color_map[g] for g in annotation]
    # update scatter
    self.update_scatter()


  def set_input_file(self, attr, old, new):
    self.active_file = new
    self.logger.info(f'Setting input file to: {new}')
    set_active_slide(self.active_file, self.shared_var, self.logger)
    for c in self.shared_var['categoricals']:
      self.logger.info(f'new annotation choice: {c}')
    self.scatter_widgets.choose_annotation.options = self.shared_var['categoricals']
    available_images = [ch for ch, val in self.shared_var['image_sources'].items() if val is not None]

    self.logger.info(f'populating available channels...')
    color_widgets = self.image_colors.default_color_widget_list()
    for ch in available_images:
      default_color = self.shared_var['channel_colors'][ch]
      self.image_colors.add_channel(ch, default_color)
      color_widgets.append(self.image_colors.get_color_widget(ch))
    self.image_colors.layout.children = color_widgets

    self.scatter_widgets.hover_tooltips.options = self.shared_var['adata_data'].columns.tolist()

    # This is a hack to pass a value through to update_scatter without introducing an arg.
    inds = np.arange(self.shared_var['n_cells'])
    self.update_box_selection(None, None, inds)
    self.reframe = True
    self.update_scatter()
    self.scatter_image.image_source.data = {
      'value': [], 'dw': [], 'dh': [], 'x0': [], 'y0': [],
      }
    # reset __everything__


  def update_scatter(self):
    """ Parse all the applicable requests and update the scatter figure
    """
    # Do the data operations, then dispatch to a method in the ImageScatterPane class
    # to handle updating the data source and redrawing
    self.logger.info('Updating scatter plot')
    self.get_selected_clusters()
    n_hl = self.shared_var['highlight_cells'].sum()

    colors = np.array(self.shared_var['color'], dtype=object)
    colors[~self.shared_var['cluster_selection']] = self.shared_var['background_color']
    sizes = np.zeros_like(colors, dtype=np.uint8)+self.scatter_widgets.dot_size.value

    self.scatter_image.p.title.text = f"Highlighting {n_hl} cells"
    self.scatter_image.scatter_source.data = dict(
      x=self.shared_var['adata_data']['coordinates_1'],
      y=self.shared_var['adata_data']['coordinates_2']-self.shared_var['y_shift'],
      s=sizes,
      index=self.shared_var['adata_data']['index_num'],
      color=colors,
    )
    if self.reframe:
      update_bbox(self.shared_var, self.scatter_image.p, self.logger)
    self.reframe=False


  def update_box_selection(self, attr, old, new):
    self.logger.info('Handling bbox selection')
    box_selection = np.ones(self.shared_var['n_cells'], dtype=bool)
    if len(new) > 0:
      box_selection = np.zeros(self.shared_var['n_cells'], dtype=bool)
      box_selection[new] = 1

    self.shared_var['box_selection'] = box_selection
    self.logger.info(f'bbox selection: {box_selection.sum()}')
    self.shared_var['highlight_cells'] = box_selection & self.shared_var['cluster_selection']
    self.reframe=True
    self.update_scatter()


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
    use_channels = [k for k,v in self.image_colors.active_channels.items() if v]
    use_channels = [k for k in use_channels if k not in ['nuclei', 'membrane']]
    if len(use_channels) == 0:
      self.logger.info('No active channels. Not drawing anything')
      return

    if sum(self.shared_var['bbox'])==0:
      self.logger.info('No bbox selected. Select a region to draw first.')
      return
    
    bbox = self.shared_var['bbox']
    bbox_plot = self.shared_var['bbox_plot']
    images = maybe_pull(use_channels, 
                        self.shared_var['active_raw_images'],
                        self.shared_var['image_sources'],
                        self.shared_var['bbox'],
                        self.image_colors.resize.value,
                        self.logger
                        )

    colors = []
    for ch in use_channels:
      c = self.image_colors.channel_color_pickers(ch).color
      chc = np.array(self.hex_to_dec(c))
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
      nuclei_color = self.hex_to_dec(nuclei_color)
      self.logger.info(f'drawing nuclei with color {nuclei_color}')
    else:
      nuclei = None
      nuclei_color = None
    blended = blend_images(images, saturation_vals=saturation, colors=colors, 
                          nuclei=nuclei, nuclei_color=nuclei_color)
    blended = blended[::-1,:] # flip
    self.logger.info(f'created blended image: {blended.shape}')

    ## Set the aspect ratio according to the selected area
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
      'x0': [bbox_plot[0]],
      'y0': [bbox_plot[2]],
      }

  # https://docs.bokeh.org/en/latest/docs/gallery/color_sliders.html
  def hex_to_dec(self, hex):
    red = ''.join(hex.strip('#')[0:2])
    green = ''.join(hex.strip('#')[2:4])
    blue = ''.join(hex.strip('#')[4:6])
    return (int(red, 16), int(green, 16), int(blue,16), 255)

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
