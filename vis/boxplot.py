import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from bokeh.io import curdoc
from bokeh.plotting import figure, output_file, show
from bokeh.models import (ColumnDataSource, Button)



class BokehBoxPlot:
  def __init__(self, data, varnames=None, height=100, width=1000): 
    self.SOURCE = ColumnDataSource(
      dict(varnames=[], q1=[], q2=[], q3=[], 
           iqr=[], upper=[], lower=[], 
           boxwidth=[], whiskw=[], whiskh=[])
    )

    self.varnames = varnames
    self.update_data(data)

    self.p = figure(title="", tools="", x_range=self.varnames, 
              plot_height=height, plot_width=width,
              toolbar_location=None, sizing_mode="scale_both")

    # stems
    self.p.segment('varnames', 'upper', 'varnames', 'q3', source=self.SOURCE, line_color="white")
    self.p.segment('varnames', 'lower', 'varnames', 'q1', source=self.SOURCE, line_color="white")

    # boxes
    self.p.vbar('varnames', 'boxwidth', 'q2', 'q3', source=self.SOURCE, fill_color="#E08E79", 
      line_color="white")
    self.p.vbar('varnames', 'boxwidth', 'q1', 'q2', source=self.SOURCE, fill_color="#E08E79", 
      line_color="white")

    # whiskers (almost-0 height rects simpler than segments)
    self.p.rect('varnames', 'lower', 'whiskw', 'whiskh', source=self.SOURCE, line_color="white")
    self.p.rect('varnames', 'upper', 'whiskw', 'whiskh', source=self.SOURCE, line_color="white")

    # # outliers
    # if not out.empty:
    #     p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

    self.p.xgrid.grid_line_color = None
    self.p.ygrid.grid_line_color = "white"
    self.p.grid.grid_line_width = 2
    self.p.xaxis.major_label_text_font_size="16px"

    self.p.yaxis.axis_label_text_font_size = '12pt'
    self.p.xaxis.major_label_orientation = np.pi/2

    self.update_boxplot()

  def update_data(self, new_data):
    new_data = new_data.loc[:, self.varnames]
    self.data = new_data

  def update_boxplot(self):
    # find the quartiles and IQR for each category
    # groups = data.groupby('groups')
    q1 = self.data.quantile(q=0.25)
    q2 = self.data.quantile(q=0.5)
    q3 = self.data.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr

    qmin = self.data.quantile(q=0.00)
    qmax = self.data.quantile(q=1.00)
    for v in self.varnames:
      if upper[v]>qmax[v]:
        upper[v] = qmax[v]
      if lower[v]<qmin[v]:
        lower[v] = qmin[v]
      

    self.SOURCE.data = dict(varnames=self.varnames,
                            q1=q1.tolist(), 
                            q2=q2.tolist(), 
                            q3=q3.tolist(), 
                            iqr=iqr.tolist(), 
                            upper=upper.tolist(), 
                            lower=lower.tolist(),
                            boxwidth=[0.7]*len(self.varnames),
                            whiskw=[0.2]*len(self.varnames),
                            whiskh=[0.01]*len(self.varnames),
                            )




# data, labels = load_iris(return_X_y=True, as_frame=True)
# varnames = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
# data.columns = varnames
# # u_labels = np.unique(labels)
# # data['groups'] = labels

# # # generate some synthetic time series for six different categories
# # cats = list("abcdef")
# # yy = np.random.randn(2000)
# # g = np.random.choice(cats, 2000)
# # for i, l in enumerate(cats):
# #     yy[g == l] += i // 2
# # df = pd.DataFrame(dict(score=yy, group=g))


# # # find the outliers for each category
# # def outliers(group):
# #     cat = group.name
# #     return group[(group.score > upper.loc[cat]['score']) | (group.score < lower.loc[cat]['score'])]['score']
# # out = groups.apply(outliers).dropna()

# # # prepare outlier data for plotting, we need coordinates for every outlier.
# # if not out.empty:
# #     outx = []
# #     outy = []
# #     for keys in out.index:
# #         outx.append(keys[0])
# #         outy.append(out.loc[keys[0]].loc[keys[1]])

# # # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
# # qmin = groups.quantile(q=0.00)
# # qmax = groups.quantile(q=1.00)
# # upper.score = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,'score']),upper.score)]
# # lower.score = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,'score']),lower.score)]


# # output_file("boxplot.html", title="boxplot.py example")

# # show(p)

# boxplot = BokehBoxPlot(data, varnames=varnames)

# curdoc().add_root(boxplot.p)