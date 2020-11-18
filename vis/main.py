# import sqlite3 as sql
from os.path import dirname, join

import numpy as np
import scanpy as sc
import pandas as pd
# import pandas.io.sql as psql

from bokeh.io import curdoc
from bokeh.layouts import column, layout
from bokeh.models import ColumnDataSource, Button, Div, Select, Slider, TextInput, MultiChoice
from bokeh.plotting import figure
# from bokeh.sampledata.movies_data import movie_path

# curdoc().theme = 'caliber'

# Use RGB color blending to show signal intensities as gradients of solid colors
def blend_colors():
  pass


ad = sc.read_h5ad("notebooks/tests/dataset.h5ad")
print(f'\n\nVisualizing {ad.shape[0]} cells\n\n')
data = ad.obs.copy()
data[ad.var_names.tolist()] = pd.DataFrame(ad.X.copy(), index=ad.obs_names, columns=ad.var_names)
data['coordinates_1'] = ad.obsm['coordinates'][:,0]
data['coordinates_2'] = ad.obsm['coordinates'][:,1]

color_map = {k: v for k, v in zip(np.unique(ad.obs.mean_leiden), ad.uns['mean_leiden_colors'])}
data['color'] = [color_map[g] for g in ad.obs.mean_leiden]
data['active_color'] = [color_map[g] for g in ad.obs.mean_leiden]

background_color = '#636363'

axis_map = {
  "coordinates_1": "coordinates_1",
  "coordinates_2": "coordinates_2"
}

desc = Div(text=open(join(dirname(__file__), "description.html")).read(), sizing_mode="stretch_width")

## Create Input controls
all_clusters = list(np.unique(ad.obs.mean_leiden))
hl_cluser_multichoice = MultiChoice(title='Focus clusters', value=[], options=all_clusters)

## Set up a way to edit the displayed colors
# edit_cluster_colors = MultiChoice(title='Focus clusters', value=[], options=all_clusters)

## 

# Create Column Data Source that will be used by the plot
source_bg = ColumnDataSource(data=dict(x=[], y=[], mean_leiden=[], z_leiden=[]))
source_fg = ColumnDataSource(data=dict(x=[], y=[], mean_leiden=[], z_leiden=[], color=[]))


# Create the figure
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
           tools='pan,wheel_zoom,reset,hover,lasso_select,save', sizing_mode="scale_both",
           tooltips=TOOLTIPS, output_backend='webgl')

# Keep a reference to the foreground set
r = p.circle(x="x", y="y", source=source_fg, radius=10, color="active_color", line_color=None)
p.circle(x="x", y="y", source=source_bg, radius=5, color=background_color, line_color=None)
p.xaxis.axis_label = 'coordinates_1'
p.yaxis.axis_label = 'coordinates_2'


# Define actions for our buttons
def select_cells():
    hl_cluster_vals = hl_cluser_multichoice.value

    if len(hl_cluster_vals) == 0:
      hl_idx = np.ones(data.shape[0], dtype='bool')
    else:
      hl_idx = data.mean_leiden.isin(hl_cluster_vals)

    hl_data = data[hl_idx]
    bg_data = data[~hl_idx]

    return hl_data, bg_data


def update():
    df_fg, df_bg = select_cells()
    # x_name = axis_map[x_axis.value]
    # y_name = axis_map[y_axis.value]
    # p.xaxis.axis_label = x_axis.value
    # p.yaxis.axis_label = y_axis.value

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


# Controls that will update the plotting area
controls = [
  hl_cluser_multichoice, 
  # x_axis, 
  # y_axis
]

for control in controls:
    control.on_change('value', lambda attr, old, new: update())

# Add a clear button for multiselect
def set_cleared_focus(*args):
  hl_cluser_multichoice.value = []
clear_button = Button(label='Clear focused cells', button_type='success')
clear_button.on_click(set_cleared_focus)
controls.append(clear_button)

inputs = column(*controls, width=320, height=1000)
inputs.sizing_mode = "fixed"
l = layout([
    [desc],
    [inputs, p],
], sizing_mode="scale_both")

print('Populating initial data')
update()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "CODEX"
