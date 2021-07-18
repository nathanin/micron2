from os.path import dirname, join

import numpy as np
import scanpy as sc
import pandas as pd
# import pandas.io.sql as psql

from bokeh.io import curdoc
from bokeh.layouts import column, row, layout, gridplot
from bokeh.models import (ColumnDataSource, BoxSelectTool, LassoSelectTool, Button, 
                          Div, Select, Slider, TextInput, MultiChoice, ColorPicker, 
                          Dropdown, Span, CheckboxButtonGroup, RangeSlider,
                          CheckboxGroup, Spinner)
from bokeh.plotting import figure
from sklearn.datasets import load_iris


# https://docs.bokeh.org/en/latest/docs/gallery/color_sliders.html
def hex_to_dec(hex):
  red = ''.join(hex.strip('#')[0:2])
  green = ''.join(hex.strip('#')[2:4])
  blue = ''.join(hex.strip('#')[4:6])
  return (int(red, 16), int(green, 16), int(blue,16), 255)

class ScatterGate:
  def __init__(self, data, height=200, width=200, 
               initial_values=['sepal_length', 'sepal_width']):
    self.data = data
    self.selected = np.ones(data.shape[0], dtype=bool)

    self.SHARED = dict(
      unselected_color = '#000000',
      selected_color =   '#ff0000',
      xvals = [],
      yvals = [],
    )
    self.SOURCES = dict(
      scatter_gate = ColumnDataSource({'x': [], 'y': [], 'color': []})
    )
    self.WIDGETS = dict(
      slider_x = RangeSlider(start=0, end=1, value=(0,0), step=0.1, title='x', css_classes=["my-widgets"]),
      slider_y = RangeSlider(start=0, end=1, value=(0,0), step=0.1, title='y', css_classes=["my-widgets"]),
      select_x = Select(title='select x', value=initial_values[0], 
                        options=self.data.columns.tolist(), css_classes=["my-widgets"]),
      select_y = Select(title='select y', value=initial_values[1], 
                        options=self.data.columns.tolist(), css_classes=["my-widgets"])
    )

    tools = 'pan,wheel_zoom,reset,hover,box_select,lasso_select'
    self.FIG = figure(title="", toolbar_location='left', tools=tools, 
                      plot_width=200, plot_height=200,
                      sizing_mode='scale_both',
                      output_backend='webgl')
    self.FIG.select(BoxSelectTool).select_every_mousemove = False
    self.FIG.select(LassoSelectTool).select_every_mousemove = False
    self.FIG.scatter(x='x', y='y', color='color', radius=0.01, 
                     source=self.SOURCES['scatter_gate'])

    self.FIG.xaxis.axis_label_text_font_size = '16pt'
    self.FIG.yaxis.axis_label_text_font_size = '16pt'
    self.FIG.xaxis.major_label_text_font_size = '12pt'
    self.FIG.yaxis.major_label_text_font_size = '12pt'

    self.px_span_lo = Span(location = 0, dimension='height', line_color='blue', line_width=3)
    self.px_span_hi = Span(location = 0, dimension='height', line_color='blue', line_width=3)
    self.py_span_lo = Span(location = 0, dimension='width', line_color='green', line_width=3)
    self.py_span_hi = Span(location = 0, dimension='width', line_color='green', line_width=3)

    self.FIG.add_layout(self.px_span_lo)
    self.FIG.add_layout(self.px_span_hi)
    self.FIG.add_layout(self.py_span_lo)
    self.FIG.add_layout(self.py_span_hi)


    self.WIDGETS['slider_x'].on_change('value_throttled', lambda a,o,n: self.update_scatter())
    self.WIDGETS['slider_y'].on_change('value_throttled', lambda a,o,n: self.update_scatter())
    self.WIDGETS['slider_x'].on_change('value', lambda a,o,n: self.update_spans())
    self.WIDGETS['slider_y'].on_change('value', lambda a,o,n: self.update_spans())

    self.WIDGETS['select_x'].on_change('value', lambda a,o,n: self.update_x())
    self.WIDGETS['select_y'].on_change('value', lambda a,o,n: self.update_y())

    self.selection = column([self.WIDGETS['select_x'], self.WIDGETS['select_y']])
    self.sliders = column(  [self.WIDGETS['slider_x'], self.WIDGETS['slider_y']])

    # self.LAYOUT = layout([
    #   [self.FIG],
    #   [selection, sliders],
    # ], 
    # # width=width, height=height
    # )

    self.update_x()
    self.update_y()
    self.update_scatter()


  def update_data(self, new_data):
    cols = new_data.columns.tolist() 

    self.data = new_data.copy()
    
    self.update_x()
    self.update_y()
    self.update_scatter()


  def update_spans(self):
    x0 = self.WIDGETS['slider_x'].value[0]
    x1 = self.WIDGETS['slider_x'].value[1]
    y0 = self.WIDGETS['slider_y'].value[0]
    y1 = self.WIDGETS['slider_y'].value[1]

    self.px_span_lo.location = x0
    self.px_span_hi.location = x1
    self.py_span_lo.location = y0
    self.py_span_hi.location = y1


  # Fix this to only change data for the changes
  def update_scatter(self):
    x0 = self.WIDGETS['slider_x'].value[0]
    x1 = self.WIDGETS['slider_x'].value[1]
    y0 = self.WIDGETS['slider_y'].value[0]
    y1 = self.WIDGETS['slider_y'].value[1]

    self.px_span_lo.location = x0
    self.px_span_hi.location = x1
    self.py_span_lo.location = y0
    self.py_span_hi.location = y1

    xvals = np.array(self.data[self.WIDGETS['select_x'].value])
    yvals = np.array(self.data[self.WIDGETS['select_y'].value])
    selected = (xvals>x0) & (xvals<x1) & (yvals>y0) & (yvals<y1)
    self.selected = selected

    colors = np.array([self.SHARED['unselected_color']] * len(xvals))
    colors[selected] = self.SHARED['selected_color']

    self.SOURCES['scatter_gate'].data = {'x': xvals, 'y': yvals, 'color': colors}


  def update_x(self):
    xcol = self.WIDGETS['select_x'].value
    xvals = np.array(self.data[xcol])
    dx = (max(xvals)-min(xvals)) / 100
    mn = min(xvals) - dx
    mx = max(xvals) + dx
    self.SHARED['xvals'] = xvals
    self.WIDGETS['slider_x'].start = mn
    self.WIDGETS['slider_x'].end = mx
    self.WIDGETS['slider_x'].value = (mn, mx)
    self.WIDGETS['slider_x'].step = dx
    self.FIG.xaxis.axis_label = xcol
    self.update_scatter()


  def update_y(self):
    ycol = self.WIDGETS['select_y'].value
    yvals = np.array(self.data[ycol])
    dy = (max(yvals)-min(yvals)) / 100
    mn = min(yvals) - dy
    mx = max(yvals) + dy
    self.SHARED['yvals'] = yvals
    self.WIDGETS['slider_y'].start = mn
    self.WIDGETS['slider_y'].end = mx
    self.WIDGETS['slider_y'].value = (mn,mx)
    self.WIDGETS['slider_y'].step = dy
    self.FIG.yaxis.axis_label = ycol
    self.update_scatter()




if __name__ == '__main__':
  data, labels = load_iris(return_X_y=True, as_frame=True)
  data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

  scatter_obj = ScatterGate(data) 

  curdoc().add_root(scatter_obj.LAYOUT)