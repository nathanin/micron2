from .codex_viewer import CodexViewer

from bokeh.io import curdoc
from bokeh.models import Div
from bokeh.layouts import layout
from os.path import dirname, join

desc = Div(text=open(join(dirname(__file__), "description.html")).read(), sizing_mode="stretch_width")
viewer = CodexViewer()

l = layout(
  [ 
    [desc], 
    [viewer.data_layout]
  ], 
  sizing_mode='scale_both'
)

curdoc().add_root(l)
curdoc().titl = "CODEX"