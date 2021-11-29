#!/usr/bin/env python

"""
Joint embedding of cells by transcriptomic profile + neighborhood 
"""

from operator import index, pos
import numpy as np
import pandas as pd
# import scanpy as sc

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch_cluster import random_walk

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler as RawNeighborSampler

from scipy.spatial import Delaunay

import tqdm.auto as tqdm
import itertools
import glob
import sys
import os

import logging
import argparse

class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super().sample(batch)


# class NeighborSampler(RawNeighborSampler):
#     def sample(self, batch):
#         batch = torch.tensor(batch)
#         row,col,_ = self.adj_t.coo()
        
#         # sample a direct neighbor (positive example) and a random node (negative example)
#         pos_batch = random_walk(row,col,batch,walk_length=1,coalesced=False)[:,1]
#         neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ), dtype=torch.long)
        
#         batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
#         return super(NeighborSampler, self).sample(batch)


class SAGE_AE(nn.Module):
    def __init__(self, in_channels, hidden_channels, output_channels, num_layers):
        super(SAGE_AE, self).__init__()
        self.num_layers = num_layers

        ae_bottleneck_channels = int(output_channels/2)

        self.ae_d1 = nn.Linear(in_channels, 256)
        self.ae_d2 = nn.Linear(256, 128)
        self.ae_d3 = nn.Linear(128, 128)
        self.ae_bottleneck = nn.Linear(128, ae_bottleneck_channels)
        self.ae_u1 = nn.Linear(ae_bottleneck_channels, 128)
        # self.ae_u2 = nn.Linear(128,128)
        self.ae_u3 = nn.Linear(128,256)
        self.ae_out = nn.Linear(256, in_channels)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = ae_bottleneck_channels if i==0 else hidden_channels
            out_channels = hidden_channels if i < num_layers-1 else output_channels 
            norm = True if i < num_layers-1 else False
            self.convs.append(SAGEConv(in_ch, out_channels, 
                                       normalize=norm, 
                                       root_weight=True))
            
    def ae(self, x):
        # This rate may be set according to the detection 
        # higher dropout rate for lower detection?
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.ae_d1(x)
        x = x.relu()
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.normalize(x)

        x = self.ae_d2(x)
        x = x.relu()
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.normalize(x)

        x = self.ae_d3(x)
        x = x.relu()
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.normalize(x)

        xb = self.ae_bottleneck(x)

        x = self.ae_u1(xb)
        x = x.relu()
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.normalize(x)

        # x = self.ae_u2(x)
        # x = x.relu()
        # x = F.dropout(x, p=0.25, training=self.training)
        # x = F.normalize(x)

        x = self.ae_u3(x)
        x = x.relu()
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.normalize(x)

        x = self.ae_out(x)

        return xb.detach(), x

    def forward(self, xin, adjs):
        x, xr = self.ae(xin)
        # x = torch.cat([x, xin], dim=1)
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers-1:
                x = x.relu()
                x = F.dropout(x, p=0.25, training=self.training)
        return x, xr
    
    def full_forward(self, xin, edge_index):
        x, xr = self.ae(xin)
        # x = torch.cat([x, xin], dim=1)
        for i,conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.25, training=self.training)
        return x, xr


def SAGE_loss(out, pos_out, neg_out):
    pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
    neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
    loss = -pos_loss - neg_loss
    return loss

def AE_loss(out_ae, x_in):
    loss = F.poisson_nll_loss(out_ae, x_in)
    return loss

def train(model, x, optimizer, optimizer_ae, train_loader, device, num_nodes):
    model.train()
    total_loss = 0
    total_ae_loss = 0
    for batch_size, n_id, adjs in train_loader:
        
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        optimizer_ae.zero_grad()
        
        out, out_ae = model(x[n_id], adjs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

        ae_loss = AE_loss(out_ae, x[n_id])
        s_loss = SAGE_loss(out, pos_out, neg_out)

        ae_loss.backward(retain_graph=True)
        s_loss.backward()

        optimizer.step()
        optimizer_ae.step()
        
        total_loss += float(s_loss) * out.size(0)
        total_ae_loss += float(ae_loss) * out.size(0)
        
    return total_loss / num_nodes, total_ae_loss / num_nodes



def point_dist(points, p1, p2):
    x1,y1 = points[p1]
    x2,y2 = points[p2]
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)



def norm_counts(x, target):
    # ref: scanpy `_normalize_data()`
    counts = np.sum(x, axis=1) 
    counts += counts == 0
    counts /= target
    np.divide(x , counts[:,None], out=x)
    return x


def read_data(src, logger=None):
    try:
        counts_srch = f'{src}/*exprMat_file.csv'
        counts_file = glob.glob(counts_srch)[0]
    except:
        if logger is not None:
            logger.info(f'{counts_srch} no matches')

    try:
        metadata_srch = f'{src}/*metadata_file.csv'
        metadata_file = glob.glob(metadata_srch)[0]
    except:
        if logger is not None:
            logger.info(f'{metadata_srch} no matches')

    if logger is not None:
        logger.info(f'reading counts from {counts_file}')
    counts = pd.read_csv(counts_file, index_col=None, header=0)
    counts.index = [f'{f}_{i}' for f,i in zip(counts.fov, counts.cell_ID)]
    usecols = [c for c in counts.columns if c not in ['fov', 'cell_ID']]
    usecols = [c for c in usecols if 'NegPrb' not in c]

    if logger is not None:
        logger.info(f'reading counts from {metadata_file}')
    metadata = pd.read_csv(metadata_file, index_col=None, header=0)
    metadata.index = [f'{f}_{i}' for f,i in zip(metadata.fov, metadata.cell_ID)]

    features = counts.loc[metadata.index, usecols].values
    points = metadata.loc[:, ['CenterX_global_px', 'CenterY_global_px']].values

    return points, features, np.array(usecols), np.array(metadata.index)



def main(ARGS, logger):

    points, features, feature_names, barcodes = read_data(ARGS.data, logger)
    logger.info(f'read data: points: {points.shape}, labels: {features.shape}')

    if ARGS.excludegenes is not None:
        logger.info(f'Exlcuding genes from {ARGS.excludegenes}')
        toss = [l.strip() for l in open(ARGS.excludegenes, 'r')]
        toss = np.array([n in toss for n in feature_names])
        features = features[:,~toss]
        feature_names = feature_names[~toss]
        logger.info(f'Using features {features.shape}')

    if ARGS.gene_detection_cutoff < 1:
        feature_detected = (features > 0).mean(axis=0)
        passed = feature_detected > ARGS.gene_detection_cutoff
        logger.info(f'Genes passed detection rate cutoff: {np.sum(passed)}')
        features = features[:, passed]
        feature_names = feature_names[passed]
        logger.info(f'Using features {features.shape}')

    if ARGS.countnorm is not None:
        logger.info(f'Normalizing counts to {ARGS.countnorm} per cell')
        features = norm_counts(features.astype(np.float32), ARGS.countnorm)
    
    if ARGS.logcounts:
        logger.info(f'Applying log1p')
        features = np.log1p(features)

    # // done setting up features

    # // Work features into a graph representation
    logger.info(f'Applying Delaunay triangulation to points')
    tri = Delaunay(points)

    logger.info(f'Building edge list from graph')
    edge_index_list = []
    for s in tqdm.tqdm(tri.simplices):
        for p1,p2 in itertools.combinations(s, 2):
            d = point_dist(points, p1,p2)
            if d > ARGS.maxdist: continue
            edge_index_list.append([p1,p2])
            edge_index_list.append([p2,p1]) # also add the reverse edge; we're undirected.

    edge_index = torch.tensor(edge_index_list, dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float32)
    logger.info(f'edge index: {edge_index.shape}')

    data = Data(x=x, edge_index=edge_index.t().contiguous())

    # // Build the model
    device = torch.device('cuda')
    model = SAGE_AE(data.num_node_features, 
                    hidden_channels=ARGS.hidden_features, 
                    output_channels=ARGS.output_features, 
                    num_layers=ARGS.num_layers)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.lr)
    optimizer_ae = torch.optim.Adam(model.parameters(), lr=ARGS.lr)

    # Start off with a big learning rate, then quickly reduce it
    scheduler = MultiStepLR(optimizer, 
                    #    step_size=ARGS.lr_step, 
                       milestones=ARGS.lr_step,
                       gamma=ARGS.lr_gamma)

    scheduler_ae = MultiStepLR(optimizer_ae, 
                    #    step_size=ARGS.lr_step, 
                       milestones=ARGS.lr_step,
                       gamma=ARGS.lr_gamma)

    x , edge_index = data.x.to(device), data.edge_index.to(device)

    train_loader = NeighborSampler(data.edge_index, 
                                   sizes=[ARGS.sampling]*ARGS.num_layers, 
                                   batch_size=ARGS.batch_size, 
                                   shuffle=True, 
                                   num_nodes=data.num_nodes)

    loss = 0
    history = []
    ae_history = []
    with tqdm.trange(ARGS.epochs) as pbar:
        for _ in pbar:
            loss, ae_loss = train(model, x, optimizer, optimizer_ae, train_loader, device, data.num_nodes)

            pbar.set_description(f'loss = {loss:3.4e} ae_loss = {ae_loss:3.4e}')

            history.append(loss)
            ae_history.append(ae_loss)

            scheduler.step()
            scheduler_ae.step()

    logger.info('Finished training')
    model = model.to(torch.device("cpu"))

    logger.info('Running forward for all cells in CPU mode')
    model.training = False
    embedded_spatial, reconstructed_counts = model.full_forward(x.cpu(), 
        edge_index.cpu())
    embedded_spatial = embedded_spatial.detach().numpy()

    embedded_cells,_ = model.ae(x.cpu())
    embedded_cells = embedded_cells.detach().numpy()

    outfbase = f'{ARGS.outprefix}-'+\
               f'{ARGS.hidden_features}hidden-'+\
               f'{ARGS.output_features}output-'+\
               f'{ARGS.num_layers}layer-'+\
               f'{ARGS.sampling}sample-'+\
               f'{ARGS.epochs}epoch'

    logger.info(f'saving to: {ARGS.outdir}/{outfbase}')

    np.save(f'{ARGS.outdir}/{outfbase}-in-situ-coords.npy', points)
    np.save(f'{ARGS.outdir}/{outfbase}-embedding-spatial.npy', embedded_spatial)
    np.save(f'{ARGS.outdir}/{outfbase}-embedding-cells.npy', embedded_cells)
    np.save(f'{ARGS.outdir}/{outfbase}-barcodes.npy', barcodes)
    np.save(f'{ARGS.outdir}/{outfbase}-feature-names.npy', feature_names)
    np.save(f'{ARGS.outdir}/{outfbase}-loss-history.npy', np.array(history))
    np.save(f'{ARGS.outdir}/{outfbase}-loss-ae-history.npy', np.array(ae_history))
    torch.save(model, f'{ARGS.outdir}/{outfbase}-model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input/output/system arguments
    parser.add_argument('data', 
        help = "Experiment directory"
    )
    parser.add_argument('-o', '--outdir', type=str, default='./out')
    parser.add_argument('--outprefix', type=str, default='districts')
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--logfile', type=str, default='log.txt')

    # Data arguments
    parser.add_argument('--maxdist', type=int, default=200)
    parser.add_argument('--logcounts', action='store_true')
    parser.add_argument('--countnorm', type=int, default=None)
    parser.add_argument('--excludegenes', type=str, default=None)
    parser.add_argument('--gene_detection_cutoff', type=float, default=0.05)

    # Model and training arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--hidden_features', type=int, default=256)
    parser.add_argument('--output_features', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--sampling', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=3096)

    parser.add_argument('--lr_step', nargs='+', type=int, default=[10])
    parser.add_argument('--lr_gamma', type=int, default=0.1)

    ARGS = parser.parse_args()

    if not os.path.exists(ARGS.outdir):
        os.makedirs(ARGS.outdir)

    logger = logging.getLogger()
    logger.setLevel('INFO')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(f'{ARGS.outdir}/{ARGS.logfile}')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info('Starting districts optimization')
    logger.info('ARGUMENTS:')
    for k,v in ARGS.__dict__.items():
        logger.info(f'\t{k}: {v}')

    main(ARGS, logger)