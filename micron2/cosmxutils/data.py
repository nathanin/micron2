import glob
import pandas as pd
import numpy as np

def read_data(src, logger, return_tables=False):
    try:
        counts_srch = f'{src}/*exprMat_file.csv'
        counts_file = glob.glob(counts_srch)[0]
    except:
        logger.info(f'{counts_srch} no matches')

    try:
        metadata_srch = f'{src}/*metadata_file.csv'
        metadata_file = glob.glob(metadata_srch)[0]
    except:
        logger.info(f'{metadata_srch} no matches')

    logger.info(f'reading counts from {counts_file}')
    counts = pd.read_csv(counts_file, index_col=None, header=0)
    counts.index = [f'{f}_{i}' for f,i in zip(counts.fov, counts.cell_ID)]

    logger.info(f'reading counts from {metadata_file}')
    metadata = pd.read_csv(metadata_file, index_col=None, header=0)
    metadata.index = [f'{f}_{i}' for f,i in zip(metadata.fov, metadata.cell_ID)]

    if return_tables:
        counts = counts.loc[metadata.index]
        return counts, metadata
    else:
        usecols = [c for c in counts.columns if c not in ['fov', 'cell_ID']]
        usecols = [c for c in usecols if 'NegPrb' not in c]
        features = counts.loc[metadata.index, usecols].values
        points = metadata.loc[:, ['CenterX_global_px', 'CenterY_global_px']].values
        return points, features, np.array(usecols), np.array(metadata.index)