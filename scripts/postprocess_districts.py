#!/usr/bin/env python

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['svg.fonttype'] = 'none'

import logging
import argparse
import json

import cuml
# from cuml.neighbors import NearestNeighbors as cu_NearestNeighbors
import tqdm.auto as tqdm
import glob
import sys

from sklearn.metrics import roc_auc_score


def run_test(x, k, seed=999, training_pct=0.75):
    np.random.seed(seed)

    clusterf = cuml.KMeans(n_clusters=k, n_init=20)
    clusterf.fit(x)
    clusters = clusterf.labels_
    cluster_levels , clusters = np.unique(clusters, return_inverse=True)

    """
    test 1-vs-rest prediction AUC
    """
    aucs = []
    coefs = []
    training = np.random.choice([0,1], clusters.shape[0])==1
    training = np.random.binomial(1,training_pct,clusters.shape[0])==1
    for test_cluster in cluster_levels:
        Xtrain_1  = x[(clusters == test_cluster)&training]
        Xtrain_2 = x[(clusters != test_cluster)&training]
        X = np.concatenate([Xtrain_1,Xtrain_2], axis=0)
        X = (X.T / X.sum(axis=1)).T
        Y = np.array([1]*Xtrain_1.shape[0] + [0]*Xtrain_2.shape[0])
        cls = cuml.LogisticRegression(penalty='elasticnet', fit_intercept=False,
                                      l1_ratio=0.5)
        cls.fit(X, Y)
        # cls = cuml.neighbors.KNeighborsClassifier(n_neighbors=100)
        # cls.fit(X, Y)

        Xtest_1  = x[(clusters == test_cluster)&~training]
        Xtest_2 = x[(clusters != test_cluster)&~training]
        Xtest = np.concatenate([Xtest_1,Xtest_2], axis=0)
        Xtest = (Xtest.T / Xtest.sum(axis=1)).T
        Ytest = np.array([1]*Xtest_1.shape[0] + [0]*Xtest_2.shape[0])
        Ypred = cls.predict_proba(Xtest)
        aucs.append(roc_auc_score(Ytest==1, Ypred[:,1]))
    
    return aucs, clusters


def makeplot(ks, avg_aucs, best_k, outf):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(ks, avg_aucs, lw=2)
    ax.set_ylabel('average AUC')
    ax.set_xlabel('k')
    ax.set_xticks(ks)
    ax.axvline(best_k, lw=1, color='k', ls='--')
    ax.set_title('district predictability')
    plt.savefig(outf, bbox_inches='tight', transparent=True)


def main(ARGS, logger):
    overall_aucs = {}
    all_cluster_vectors = {}

    embedding_file = glob.glob(f'{ARGS.homedir}/*embedding.npy')[0]
    barcode_file = glob.glob(f'{ARGS.homedir}/*barcodes.npy')[0]

    x = np.load(embedding_file)
    barcodes = np.load(barcode_file, allow_pickle=True)

    logger.info(f'Loaded x = {x.shape}, barcodes = {barcodes.shape}')
    for k in tqdm.tqdm(ARGS.ktest):
        aucs, clusters = run_test(x, k, seed=ARGS.seed, training_pct=ARGS.training_pct)
        overall_aucs[k] = aucs.copy()
        all_cluster_vectors[k] = clusters.copy()

    ks = []
    avg_aucs = []
    for k,v in overall_aucs.items():
        ks.append(k)
        avg_aucs.append(np.mean(v))

    best_idx = np.argmax(avg_aucs)
    best_k = ks[best_idx]
    logger.info(f'best k: {best_k}')
    logger.info(f'best AUC= {avg_aucs[best_idx]}')

    outdir = ARGS.homedir if ARGS.outdir is None else ARGS.outdir

    plotf = f'{outdir}/aucs.svg'
    makeplot(ks, avg_aucs, best_k, plotf)

    best_clusters = all_cluster_vectors[best_k]

    df = pd.DataFrame(index=barcodes, columns=['cluster'])
    df['cluster'] = pd.Categorical(best_clusters)
    tablef = f'{outdir}/clusters.csv'
    df.to_csv(tablef)

    reportf = f'{outdir}/postprocess_report.json'
    with open(reportf, 'w+') as f:
        json.dump({
            'best_k': int(best_k),
            'best_auc': float(avg_aucs[best_idx]),
            'best_idx': int(best_idx),
            'aucs': list(avg_aucs),
            'tested_ks': list(ks)
        }, f,
        indent='\t')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('homedir')
    parser.add_argument('--log', type=str, default='postprocess_log.txt')
    parser.add_argument('--ktest', nargs='+', type=int, 
        default=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,17,18,19,20]
    )
    parser.add_argument('-o', '--outdir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--training_pct', type=float, default=0.75)

    ARGS = parser.parse_args()
    outdir = ARGS.homedir if ARGS.outdir is None else ARGS.outdir

    logger = logging.getLogger()
    logger.setLevel('INFO')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(f'{outdir}/{ARGS.log}')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info('Starting districts postprocessing')
    logger.info('ARGUMENTS:')
    for k,v in ARGS.__dict__.items():
        logger.info(f'\t{k}: {v}')


    main(ARGS, logger)