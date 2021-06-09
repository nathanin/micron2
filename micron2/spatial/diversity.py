"""
Diversity functions originally by K.H.G (@gouink) 2021

""" 

import numpy as np

__all__ = [
  'gini',
  'shannon'
]
def gini(data):
  """ calculate Gini coefficient

  from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
  
  Args:
    data (pd.Series)
  
  Returns:
    gini (float)
  """
  vals = data.value_counts()
  vals = vals[vals>0].to_numpy()

  vals = vals.flatten() #all values are treated equally, tcrs must be 1d 
  vals = np.sort(vals) #values must be sorted
  index = np.arange(1,vals.shape[0]+1) #index per tcrs element
  n = vals.shape[0] #number of tcrs elements
  
  return ((np.sum((2 * index - n  - 1) * vals)) / (n * np.sum(vals))) #Gini coefficient



def shannon(data, clonality=False):
  """ calculate normalized shannon entropy (or clonality)

  clonality = 1 - (SE/ln(# of species))

  Args:
    data (pd.Series)
  Returns:
    entropy (float)
  """
  totalsize = len(data)

  vals = data.value_counts()
  vals = vals/totalsize
  numel = len(vals)
  
  runningSum = 0
  
  for i in vals:
      runningSum += -1*(i * np.log(i))

  if clonality:
    return (1 - (runningSum/np.log(numel)))
  else:
    return runningSum/np.log(numel)