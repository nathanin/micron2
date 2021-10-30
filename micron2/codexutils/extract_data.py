import numpy as np
import tifffile import imread as tif_imread
import zarr

from micron2.codexutils import get_images


def gen_cells(coords, image_src, mask, size=128):
    z = tif_imread(image_src, aszarr=True)
    image_store = zarr.open(z, mode='r')

    for coord in coords:
        r,c = coord
        b = [r-size, r+size, c-size, c+size]
        mask_part = mask[b[0]:b[1], b[2]:b[3]]
        img = image_store[b[0]:b[1], b[2]:b[3]]
        m = mask_part == mask_part[size,size]
        yield img, m


def mean_cell(v, m):
    return np.mean(v[m])

def quantile_cell(v, m, q=0.75):
    return np.quantile(v[m], q)


def make_cell_intensity_table(coords, mask, channels, image_sources, agg_fn=mean_cell, agg_fn_kwargs={}):
    """
    channels: a list (['PanCytoK', 'CD45', 'CD3e])
    image_sources: a dict, all([ch in image_sources.keys() for ch in channels])
        {
            'PanCytoK': '/data/sample/pancytok.tif',
            'CD45': '/data/sample/cd45.tif',
            'CD3e': '/data/sample/cd3e.tif',
        }
    agg_fn: a function whose call accepts an image and mask as the first 2 arguments
            followed by keword arguments
    """
    values = np.zeros((len(coords), len(channels)), dtype=np.float32)
    for i, ch in enumerate(channels):
        src = image_sources[ch]
        for j, cell_mask in enumerate(gen_cells(coords, src, mask): 
            v, m = cell_mask
            values[i,j] = agg_fn(v,m,**agg_fn_kwargs)

    return values


