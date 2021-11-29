#!/usr/bin/env python

"""
QC reads in a slide (collections of FOVs)
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns

import glob
import argparse
import logging

import os
import sys
import shutil

from data import read_data
import scanpy as sc
from anndata import AnnData

from scipy.sparse import csr_matrix



def cell_counts_plots(expr, metadata, targets, negprobes, outf):
    cpcm = pd.DataFrame(expr.loc[metadata.index,targets].apply(lambda x: np.mean(x), axis=1), columns=['mean_counts'])
    cpct = pd.DataFrame(expr.loc[metadata.index,targets].apply(lambda x: np.sum(x), axis=1), columns=['total_counts'])
    cpcmn = pd.DataFrame(expr.loc[metadata.index,negprobes].apply(lambda x: np.mean(x), axis=1), columns=['mean_counts_negprb'])
    cpctn = pd.DataFrame(expr.loc[metadata.index,negprobes].apply(lambda x: np.sum(x), axis=1), columns=['total_counts_negprb'])

    cell_qc = pd.concat([cpcm, cpct, cpcmn, cpctn], axis=1)
    cell_qc['fov'] = pd.Categorical(metadata['fov'].values)

    fig = plt.figure(figsize=(10,8), dpi=180)
    ax = fig.add_subplot(4,1,1)
    sns.boxplot(data=cell_qc, x='fov', y='mean_counts', ax=ax, fliersize=1)

    ax = fig.add_subplot(4,1,2)
    sns.boxplot(data=cell_qc, x='fov', y='total_counts', ax=ax, fliersize=1)

    ax = fig.add_subplot(4,1,3)
    sns.boxplot(data=cell_qc, x='fov', y='mean_counts_negprb', ax=ax, fliersize=1)

    ax = fig.add_subplot(4,1,4)
    sns.boxplot(data=cell_qc, x='fov', y='total_counts_negprb', ax=ax, fliersize=1)

    plt.subplots_adjust(wspace=0.5)
    plt.savefig(outf, transparent=True, bbox_inches='tight')
    plt.close()
    return cell_qc


def detection_plot(expr, outf):
    nz = pd.DataFrame(expr.apply(lambda x: np.log10(1+np.sum(x>0)), axis=0), columns=['nonzero_cells'])
    tot = pd.DataFrame(expr.apply(lambda x: np.log10(1+np.sum(x)), axis=0), columns=['total_counts'])

    # fig = plt.figure(figsize=(6,3), dpi=90)
    # ax = fig.add_subplot(2,2,1)
    fig, axs = plt.subplots(figsize=(9,4), ncols=2, nrows=2)
    gs = axs[1,1].get_gridspec()
    for ax in axs[-1,:]:
        ax.remove()

    axs = axs.ravel()

    r = tot.values[:,0]/nz.values[:,0]

    ax = axs[0]
    _ = ax.hist(r, bins=50, log=False)
    ax.set_ylabel('targets')
    ax.set_xlabel('Total counts / N cells')
    z = np.quantile(r, 0.99) # 99% ~ 10 genes / 1000
    ax.axvline(z, lw=1, ls='--', color='k')

    outliers = tot[r>z].index
    print(len(outliers))

    ax = axs[1]
    # ax = fig.add_subplot(2,2,2)
    ax.scatter(x=nz.values, y=tot.values, s=0.5)
    ax.plot([3.5,6.5], [3.5,6.5], lw=1, ls='--', c='k')
    ax.set_xlabel('N cells (log10(1+x))')
    ax.set_ylabel('total counts (log10(1+x))')

    for g in outliers:
        ax.scatter(nz.T[g], tot.T[g], color='r', s=3)
        
    # ax = fig.add_subplot(2,2,3)
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_and_subplots.html
    ax = fig.add_subplot(gs[-1,:])
    nz.loc[outliers,:].sort_values('nonzero_cells', ascending=True).plot.bar(y='nonzero_cells', ax=ax)

    plt.savefig(outf, transparent=True, bbox_inches='tight')
    plt.close()
    return outliers


ALPHABET = list('abcdefghijklmnopqrstuvwxyz1234567890')
def main(ARGS, logger):
    salt = ''.join(np.random.choice(ALPHABET,5,replace=True))

    logger.info(f'Loading from {ARGS.data}')
    expr, metadata = read_data(ARGS.data, logger, return_tables=True)
    logger.info(f'expr: {expr.shape}')
    logger.info(f'metadata: {metadata.shape}')

    expr.index = [f'{f}_{i}' for f,i in zip(expr.fov, expr.cell_ID)]
    metadata.index = [f'{f}_{i}' for f,i in zip(metadata.fov, metadata.cell_ID)]

    allprobes = [c for c in expr.columns if c not in ['fov', 'cell_ID']]
    targets = [c for c in allprobes if 'NegPrb' not in c]
    negprobes = [c for c in allprobes if 'NegPrb' in c]

    expr = expr.loc[metadata.index, :]

    logger.info(f'gene targets: {len(targets)} negative probes: {len(negprobes)}')


    outf = f'{ARGS.outdir}/{ARGS.outprefix}-cellQC-counts.svg'
    cell_qc = cell_counts_plots(expr, metadata, targets, negprobes, outf)
    logger.info(f'cell QC: {cell_qc.shape} --> {outf}')

    logger.info(f'Making AnnData object')
    x_sp = csr_matrix(expr.loc[expr.index, allprobes].values)
    ad = AnnData(X=x_sp, obs=metadata, 
                 var=pd.DataFrame(index=allprobes),
                 obsm = {
                     'local_coords': metadata.loc[:, ['CenterX_local_px', 'CenterY_local_px']].values,
                     'global_coords': metadata.loc[:, ['CenterX_global_px', 'CenterY_global_px']].values,
                 }
                 )
    ad.obs['fov'] = pd.Categorical(ad.obs['fov'].values)
    logger.info(f'adata: {ad.shape}')

    outf = f'{ARGS.outdir}/{ARGS.outprefix}-probeQC-counts.svg'
    overrepresented_genes = detection_plot(expr.loc[:,targets], outf)

    outf = f'{ARGS.outdir}/{ARGS.outprefix}-excl-genes.txt'
    with open(outf, 'w+') as f:
        for l in overrepresented_genes:
            f.write(f'{l}\n')

    # Making more figures 
    # FOV image
    fig = plt.figure(figsize=(4,4), dpi=180)
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    sc.pl.embedding(ad, basis='global_coords', color='fov', ax=ax, legend_loc='on data', 
        save = f'_{salt}_FOVs.png')
    outf = f'{ARGS.outdir}/{ARGS.outprefix}-FOVs.png'
    shutil.copyfile(f'.scanpy_figures/global_coords_{salt}_FOVs.png', outf)

    # Negative probe detection
    plt.rcParams['figure.dpi'] = 300
    sc.pl.dotplot(ad, negprobes, groupby='fov', standard_scale='var', 
                  save = f'_{salt}_negprobes.svg', )
    outf = f'{ARGS.outdir}/{ARGS.outprefix}-negprobes.svg'
    shutil.copyfile(f'.scanpy_figures/dotplot__{salt}_negprobes.svg', outf)

    outf = f'{ARGS.outdir}/{ARGS.outprefix}.h5ad'
    ad.write(outf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', 
        help = "Experiment directory"
    )
    parser.add_argument('-o', '--outdir', type=str, default='./out')
    parser.add_argument('--outprefix', type=str, default='Sample')

    parser.add_argument('--log', type=str, default='qc.log')

    ARGS = parser.parse_args()
    if not os.path.exists(ARGS.outdir):
        os.makedirs(ARGS.outdir)

    logger = logging.getLogger()
    logger.setLevel('INFO')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(f'{ARGS.outdir}/{ARGS.log}')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info('Starting QC')
    logger.info('ARGUMENTS:')
    for k,v in ARGS.__dict__.items():
        logger.info(f'\t{k}: {v}')

    figdir = f'.scanpy_figures'
    if not os.path.exists(figdir):
        os.makedirs(figdir)

    logger.info(f'Setting figdir to: {figdir}')
    sc.settings.figdir = figdir


    main(ARGS, logger)