#!/usr/bin/env python

import numpy as np
import pandas as pd
import scanpy as sc

from matplotlib import pyplot as plt
import seaborn as sns

import tqdm.auto as tqdm
import itertools
import shutil
import glob
import sys
import os

import logging
import argparse

import cugraph
import cuml
from cuml import UMAP 
import cudf

def leiden_GPU(X, n_neighbors, resolution): 
    x_cudf = cudf.DataFrame(X)
    model = cuml.neighbors.NearestNeighbors(n_neighbors=n_neighbors, output_type='numpy')
    model.fit(X)
    kn_graph = model.kneighbors_graph(x_cudf)

    Gin = cugraph.Graph()
    offsets = cudf.Series(kn_graph.indptr)
    indices = cudf.Series(kn_graph.indices)
    Gin.from_cudf_adjlist(offsets, indices, None)
        
    parts, modularity_score = cugraph.leiden(Gin, resolution=resolution)
    clusters = parts.sort_values('vertex').partition.values.get() # parts is a cuDF dataframe

    return clusters


def truncnorm(x, qs=[0.01, 0.99]):
    x=x.copy()
    for i in range(x.shape[1]):
        z = x[:,i]
        q1, q2 = np.quantile(z,qs)
        x[z<q1,i] = q1
        x[z>q2,i] = q2
        x[:,i] = (z-q1) / (q2 - q1)
    return x


def load_embeddings(fbase):
    embf = glob.glob(  f'{fbase}*embedding-spatial.npy')[0]
    embcf = glob.glob( f'{fbase}*embedding-cells.npy')[0]
    coordf = glob.glob(f'{fbase}*in-situ-coords.npy')[0]
    histf = glob.glob( f'{fbase}*loss-history.npy')[0]
    histcf = glob.glob(f'{fbase}*loss-ae-history.npy')[0]

    lst = {
        'embf': embf, 
        'embcf': embcf, 
        'coordf': coordf, 
        'histf': histf, 
        'histcf': histcf
    }

    return lst


def main(ARGS, logger):
    fbase = f'{ARGS.district_dir}/{ARGS.sample}-'
    logger.info(f'Loading embeddings from districts dir: {fbase}')
    sourcefiles = load_embeddings(fbase)

    coords = np.load(sourcefiles['coordf'])
    print(coords.shape)

    gs_emb = np.load(sourcefiles['embf'])
    print(gs_emb.shape)

    ae_emb = np.load(sourcefiles['embcf'])
    print(ae_emb.shape)

    # // Run UMAPs
    emb_ae = UMAP(n_neighbors=20).fit_transform(ae_emb)
    clusters_ae = leiden_GPU(ae_emb, 20, 0.5)
    nclust = len(np.unique(clusters_ae))
    logger.info(f'Cell clusters: {nclust}')

    emb_sp = UMAP(n_neighbors=20).fit_transform(gs_emb)
    clusters_sp = leiden_GPU(gs_emb, 20, 0.3)
    nclust = len(np.unique(clusters_ae))
    logger.info(f'Spatial clusters: {nclust}')

    f = f'{fbase}-umap-spatial.npy'
    np.save(f, emb_sp)

    f = f'{fbase}-umap-cells.npy'
    np.save(f, emb_ae)

    f = f'{fbase}-clusters-spatial.npy'
    np.save(f, clusters_sp)

    f = f'{fbase}-clusters-cells.npy'
    np.save(f, clusters_ae)


    # // load h5ad formatted dataset
    adata_file = f'{ARGS.qc_dir}/{ARGS.sample}.h5ad'
    ad = sc.read_h5ad(adata_file)

    ad.obs['cell_clusters'] = pd.Categorical(clusters_ae)
    ad.obs['spatial_clusters'] = pd.Categorical(clusters_sp)
    ad.obsm['X_umap_cell'] = truncnorm(emb_ae, qs=[0.01, 0.99])
    ad.obsm['X_umap_spatial'] = truncnorm(emb_sp, qs=[0.01, 0.99])

    fsave = f'{fbase}-dataset.h5ad'
    logger.info(f'Writing h5ad to {fsave}')
    ad.write(fsave)


    # // draw in situ cell type
    fig = plt.figure(figsize=(8,7), dpi=300)
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    ax.set_facecolor((1,1,1,1))
    sc.pl.embedding(ad, basis='global_coords',  
        color='cell_clusters', ax=ax, s=10, save='_cell_clusters.png')
    fsave = f'{fbase}-cell_clusters_insitu.png'
    logger.info(f'Moving output to {fsave}')
    shutil.copyfile('.scanpy_figures/global_coords_cell_clusters.png', fsave)
    plt.close()

    # // draw UMAP cell type
    fig = plt.figure(figsize=(5,5), dpi=1280)
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    ax.set_facecolor((1,1,1,1))
    sc.pl.embedding(ad, basis='X_umap_cell',  
        color='cell_clusters', ax=ax, s=10, save='_cell_clusters.png')
    fsave = f'{fbase}-cell_clusters_umap.png'
    logger.info(f'Moving output to {fsave}')
    shutil.copyfile('.scanpy_figures/X_umap_cell_cell_clusters.png', fsave)
    plt.close()

    # // draw in situ spatial-clusters
    fig = plt.figure(figsize=(8,7), dpi=300)
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    ax.set_facecolor((1,1,1,1))
    sc.pl.embedding(ad, basis='global_coords',  
        color='spatial_clusters', ax=ax, s=10, save='_spatial_clusters.png')
    fsave = f'{fbase}-spatial_clusters_insitu.png'
    logger.info(f'Moving output to {fsave}')
    shutil.copyfile('.scanpy_figures/global_coords_spatial_clusters.png', fsave)
    plt.close()

    # // draw UMAP spatial-clusters
    fig = plt.figure(figsize=(5,5), dpi=1280)
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    ax.set_facecolor((1,1,1,1))
    sc.pl.embedding(ad, basis='X_umap_spatial',  
        color='spatial_clusters', ax=ax, s=10, save='_spatial_clusters.png')
    fsave = f'{fbase}-spatial_clusters_umap.png'
    logger.info(f'Moving output to {fsave}')
    shutil.copyfile('.scanpy_figures/X_umap_cell_spatial_clusters.png', fsave)
    plt.close()


    # // run diffex on cell types
    sc.tl.rank_genes_groups(ad, 'cell_clusters', pts=True, 
                        key_added=None, copy=False, 
                        method='wilcoxon', 
                        corr_method='benjamini-hochberg', 
                        tie_correct=False)

    dfs = []
    for grp in ad.obs.cell_clusters.cat.categories:
        df = sc.get.rank_genes_groups_df(ad, group=str(grp))
        df['group'] = grp
        dfs.append(df.copy())
        
    d_genes = pd.concat(dfs, axis=0)
    logger.info('Saving differential gene expression table')
    d_genes.to_csv(f'{fbase}-cell_clusters_dgex.csv')


    marker_genes = [l.strip() for l in open(ARGS.marker_genes, 'r')]
    sc.pl.dotplot(ad, marker_genes, groupby='cell_clusters', 
        dot_max=0.8, dot_min=0.05, standard_scale='var',
        save='_general_markers.svg')

    fsave = f'{fbase}-general_markers.svg'
    shutil.copyfile('.scanpy_figures/dotplot__general_markers.svg', fsave)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input/output/system arguments
    # parser.add_argument('data', 
    #     help = "Experiment directory"
    # )
    # parser.add_argument('-o', '--outdir', type=str, default='./out')
    # parser.add_argument('--outprefix', type=str, default='Sample')

    parser.add_argument('-d', '--district_dir', type=str, default=None)
    parser.add_argument('-q', '--qc_dir', type=str, default=None)
    parser.add_argument('-s', '--sample', type=str, default=None)

    parser.add_argument('-m', '--marker_genes', type=str, default=None)

    # parser.add_argument('--logfile', type=str, default='process_districts.log')

    ARGS = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel('INFO')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh = logging.FileHandler(f'{ARGS.outdir}/{ARGS.logfile}')
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info('Starting districts optimization')
    logger.info('ARGUMENTS:')
    for k,v in ARGS.__dict__.items():
        logger.info(f'\t{k}: {v}')
        
    figdir = f'.scanpy_figures'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    logger.info(f'Setting figdir to: {figdir}')
    sc.settings.figdir = figdir

    main(ARGS, logger)