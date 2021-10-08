#!/usr/bin/env python

from tifffile import imread as tif_imread
import cv2
import numpy as np

import glob
import argparse

import os


def img2uint(img):
  q = np.quantile(img, 0.9999)
  img[img > q] = q
  img = img / q
  img = (img * 255).astype(np.uint8)
  return img


def main(inputs, outdir, resize):
  print(f'{outdir}')
  print(f'Got {len(inputs)} inputs')

  for inp in inputs:
    img = tif_imread(inp)
    print(f'{inp} {img.shape}')
    img = img2uint(img)

    if resize != 1:
      img = cv2.resize(img, dsize=(0,0), fx=resize, fy=resize)

    bn = os.path.basename(inp).replace('.tif', '.png') 
    outf = f'{outdir}/{bn}'
    print(outf)
    cv2.imwrite(outf, img)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--outdir', required=True)
  parser.add_argument('--inputs', nargs='+', required=True)
  parser.add_argument('--resize', default=0.5, type=float, required=False)

  ARGS = parser.parse_args()
  main(ARGS.inputs, ARGS.outdir, ARGS.resize)

