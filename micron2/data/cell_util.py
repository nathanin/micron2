import cv2
import numpy as np

kernel = np.ones((3,3))
# def staining_border_nonzero(h5f, channel, i, kernel=kernel):
  # x = h5f['cells'][channel][i,...]
  # m = h5f['meta']['nuclear_masks'][i,...]

def staining_border_nonzero(x, m, kernel=kernel):
  """ Approximate ring coverage with the percent of non-zero border pixels """
  md = cv2.dilate(m.astype(np.uint8),kernel,2)
  me = cv2.erode(m.astype(np.uint8),kernel,1)
  border_signal=x[(md-me)>0]
  return np.sum(border_signal > 0)/len(border_signal)