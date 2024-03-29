#!/usr/bin/env python

from micron2.data import pull_nuclei
from micron2.data import load_as_anndata
import pandas as pd
import pytiff
import glob
import h5py
import cv2
import os

import argparse



"""
Create an hdf5 dataset from source images and nuclei segmentations

Only pytiff (?) , pandas, and h5py need to be installed for this to work,
so warnings about TensorFlow, TensorFlow-io and RAPIDS.ai importing can be
ingnored
"""

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datahome', type=str, required=True)
parser.add_argument('-s', '--sample_id', type=str, required=True)
parser.add_argument('-r', '--region_num', type=int, required=True)

parser.add_argument('--size', default=64, type=int)
parser.add_argument('--min_area', default=100, type=int)
parser.add_argument('--tile_size', default=128, type=int)
parser.add_argument('--overlap', default=0.2, type=float)
parser.add_argument('--tile_scale_factor', default=1., type=float)
parser.add_argument('--debug', action='store_true')


ARGS=parser.parse_args()

full_sample_id = f'{ARGS.sample_id}_reg{ARGS.region_num}'

cells = pd.read_csv(f'{ARGS.datahome}/{full_sample_id}/{full_sample_id}_2_centroids.csv', index_col=0, header=0)
nuclei_img        = f'{ARGS.datahome}/{full_sample_id}/{full_sample_id}_2_nuclei.tif'
membrane_img      = f'{ARGS.datahome}/{full_sample_id}/{full_sample_id}_2_membrane.tif'
tissue_img        = f'{ARGS.datahome}/{full_sample_id}/{full_sample_id}_1_tissue.tif'

imagefs = sorted(glob.glob(f'{ARGS.datahome}/{full_sample_id}/images/*.tif'))
dapi_images = [f for f in imagefs if 'DAPI' in f]
non_dapi_images = [f for f in imagefs if 'DAPI' not in f]
non_dapi_images = [f for f in non_dapi_images if 'Blank' not in f]
non_dapi_images = [f for f in non_dapi_images if 'Empty' not in f]

channel_names = [os.path.basename(x) for x in non_dapi_images]
channel_names = [x.replace(f'.tif','') for x in channel_names]
channel_names = [x.split('_')[-2] for x in channel_names]
channel_names = ["DAPI"] + channel_names
print(len(channel_names))

image_paths = [dapi_images[0]] + non_dapi_images
print(len(image_paths))

out_file = f'{ARGS.datahome}/{full_sample_id}/{full_sample_id}_v6.hdf5'

pull_nuclei(cells, 
            image_paths, 
            out_file=out_file, 
            nuclei_img=nuclei_img,
            membrane_img=membrane_img,
            tissue_img=tissue_img,
            size=ARGS.size,
            min_area=ARGS.min_area, 
            tile_size=ARGS.tile_size,
            channel_names=channel_names,
            overlap=ARGS.overlap,
            tile_scale_factor=ARGS.tile_scale_factor,
            skip_tiles=True,
            debug=ARGS.debug
           )

print(f'create dataset at {out_file}: success')
