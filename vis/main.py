# import sqlite3 as sql
from os.path import dirname, join

import numpy as np
import scanpy as sc
import pandas as pd
# import pandas.io.sql as psql

from bokeh.io import curdoc
from bokeh.layouts import column, layout
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput
from bokeh.plotting import figure
# from bokeh.sampledata.movies_data import movie_path


curdoc().theme = 'caliber'

# conn = sql.connect(movie_path)
# query = open(join(dirname(__file__), 'query.sql')).read()
# movies = psql.read_sql(query, conn)


## All this is just setting up the data frame
# movies["color"] = np.where(movies["Oscars"] > 0, "orange", "grey")
# movies["alpha"] = np.where(movies["Oscars"] > 0, 0.9, 0.25)
# movies.fillna(0, inplace=True)  # just replace missing values with zero
# movies["revenue"] = movies.BoxOffice.apply(lambda x: '{:,d}'.format(int(x)))

# with open(join(dirname(__file__), "razzies-clean.csv")) as f:
#     razzies = f.read().splitlines()
# movies.loc[movies.imdbID.isin(razzies), "color"] = "purple"
# movies.loc[movies.imdbID.isin(razzies), "alpha"] = 0.9

ad = sc.read_h5ad("notebooks/tests/dataset.h5ad")
print(f'\n\nVisualizing {ad.shape[0]} cells\n\n')
data = ad.obs.copy()
data[ad.var_names.tolist()] = pd.DataFrame(ad.X.copy(), index=ad.obs_names, columns=ad.var_names)
data['coordinates_1'] = ad.obsm['coordinates'][:,0]
data['coordinates_2'] = ad.obsm['coordinates'][:,1]

color_map = {k: v for k, v in zip(np.unique(ad.obs.mean_leiden), ad.uns['mean_leiden_colors'])}
data['color'] = [color_map[g] for g in ad.obs.mean_leiden]
data['active_color'] = [color_map[g] for g in ad.obs.mean_leiden]

background_color = '#d4d4d4'

axis_map = {
  "coordinates_1": "coordinates_1",
  "coordinates_2": "coordinates_2"
}

desc = Div(text=open(join(dirname(__file__), "description.html")).read(), sizing_mode="stretch_width")

## Create Input controls
# reviews = Slider(title="Minimum number of reviews", value=80, start=10, end=300, step=10)
# min_year = Slider(title="Year released", start=1940, end=2014, value=1970, step=1)
# max_year = Slider(title="End Year released", start=1940, end=2014, value=2014, step=1)
# oscars = Slider(title="Minimum number of Oscar wins", start=0, end=4, value=0, step=1)
# boxoffice = Slider(title="Dollars at Box Office (millions)", start=0, end=800, value=0, step=1)

# size = Select(title="Cluster", value="DAPI",
#                  options=ad.var_names.tolist()))
# color = Select(title="Cluster", value="DAPI",
#                  options=ad.var_names.tolist()))

n_highlight = 4
hl_cluster_select = []
for j in range(n_highlight):
  hl_cluster_select.append( Select(title=f"Focus Cluster {j}", value="All",
                 options=list(np.unique(ad.obs.mean_leiden))+['All'])
  )
# hl_cluster_1 = Select(title="Focus Cluster 1", value="All",
#                  options=list(np.unique(ad.obs.mean_leiden)))
# hl_cluster_2 = Select(title="Focus Cluster 2", value="All",
#                  options=list(np.unique(ad.obs.mean_leiden)))


## Options for when we have cell type annotations
# genre = Select(title="Cell type", value="All",
#                options=list(np.unique(ad.obs.CellType)))
# cell_type = TextInput(title="Cell type name contains")
# director = TextInput(title="Director name contains")
# cast = TextInput(title="Cast names contains")
x_axis = Select(title="X Axis", options=sorted(axis_map.keys()), value="coordinates_1")
y_axis = Select(title="Y Axis", options=sorted(axis_map.keys()), value="coordinates_2")

# Create Column Data Source that will be used by the plot
source_bg = ColumnDataSource(data=dict(x=[], y=[]))
source_fg = ColumnDataSource(data=dict(x=[], y=[], color=[]))

# TOOLTIPS=[
#     ("Title", "@title"),
#     ("Year", "@year"),
#     ("$", "@revenue")
# ]
# p = figure(plot_height=600, plot_width=700, title="", toolbar_location=None, 
#            tooltips=TOOLTIPS, sizing_mode="scale_both")
# p.circle(x="x", y="y", source=source, size=7, color="color", line_color=None, fill_alpha="alpha")

# Draw a canvas with an appropriate aspect ratio
dx = np.abs(data.coordinates_1.max() - data.coordinates_1.min())
dy = np.abs(data.coordinates_2.max() - data.coordinates_2.min())
width = 700
height = int(width * (dy/dx))

p = figure(plot_height=height, plot_width=width, title="", toolbar_location='left', 
           tools='pan,wheel_zoom,reset,save', sizing_mode="scale_both",
           output_backend='webgl')

p.circle(x="x", y="y", source=source_bg, radius=7, color=background_color, line_color=None)
p.circle(x="x", y="y", source=source_fg, radius=15, color="active_color", line_color=None)


def select_cells():

    hl_cluster_vals = [
      hl_cluster.value for hl_cluster in hl_cluster_select
    ]

    if ['All']*len(hl_cluster_vals) == hl_cluster_vals:
      hl_idx = np.ones(data.shape[0], dtype='bool')
    else:
      hl_idx = data.mean_leiden.isin(hl_cluster_vals)

    hl_data = data[hl_idx]
    bg_data = data[~hl_idx]

    return hl_data, bg_data


def update():
    df_fg, df_bg = select_cells()
    x_name = axis_map[x_axis.value]
    y_name = axis_map[y_axis.value]

    p.xaxis.axis_label = x_axis.value
    p.yaxis.axis_label = y_axis.value

    p.title.text = "Highlighting %d cells" % len(df_fg)
    source_fg.data = dict(
        x=df_fg[x_name],
        y=df_fg[y_name],
        active_color=df_fg["active_color"],
    )
    source_bg.data = dict(
        x=df_bg[x_name],
        y=df_bg[y_name],
    )

controls = hl_cluster_select+[x_axis, y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

inputs = column(*controls, width=320, height=1000)
inputs.sizing_mode = "fixed"
l = layout([
    [desc],
    [inputs, p],
], sizing_mode="scale_both")

update()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "CODEX"
