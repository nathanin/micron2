#!/usr/bin/env python

import numpy as np
import pandas as pd
import scanpy as sc

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch_cluster import random_walk

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.data import NeighborSampler as RawNeighborSampler

from scipy.spatial import Delaunay

import tqdm.auto as tqdm
import itertools
import sys
import os

import logging
import argparse


class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row,col,_ = self.adj_t.coo()
        
        # sample a direct neighbor (positive example) and a random node (negative example)
        pos_batch = random_walk(row,col,batch,walk_length=1,coalesced=False)[:,1]
        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ), dtype=torch.long)
        
        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super(NeighborSampler, self).sample(batch)


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, output_channels, num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i==0 else hidden_channels
            out_channels = hidden_channels if i < num_layers-1 else output_channels 
            norm = True if i < num_layers-1 else False
            self.convs.append(SAGEConv(in_channels, out_channels, 
                                       normalize=norm, 
                                       root_weight=True))
            
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers-1:
                x = x.relu()
                x = F.dropout(x, p=0.25, training=self.training)
        return x
    
    def full_forward(self, x, edge_index):
        for i,conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


def train(model, x, optimizer, train_loader, device, num_nodes):
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        
        out = model(x[n_id], adjs)
        out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
        
        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss) * out.size(0)
        
    return total_loss / num_nodes

def point_dist(points, p1, p2):
    x1,y1 = points[p1]
    x2,y2 = points[p2]
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def read_data(src):
    df = pd.read_csv(src, index_col=0, header=0)
    points = np.array(df[['x', 'y']].values)
    labels_src = np.array(df['label'])
    _, labels = np.unique(labels_src, return_inverse=True)
    return points, labels, np.array(df.index)

def main(ARGS, logger):

    points, labels, barcodes = read_data(ARGS.data)

    logger.info(f'read data: points: {points.shape}, labels: {labels.shape}')
    n_labels = len(np.unique(labels))
    labels_onehot = np.eye(n_labels)[labels]
    features = labels_onehot.copy()
    logger.info(f'unique labels: {n_labels}')

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

    device = torch.device('cuda')
    model = SAGE(data.num_node_features, hidden_channels=128, output_channels=3, num_layers=3)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.lr)
    scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
    x , edge_index = data.x.to(device), data.edge_index.to(device)

    train_loader = NeighborSampler(data.edge_index, sizes=[2,2,2], batch_size=3096, 
        shuffle=True, num_nodes=data.num_nodes)

    loss = 0
    history = []
    with tqdm.trange(ARGS.epochs) as pbar:
        for _ in pbar:
            loss = train(model, x, optimizer, train_loader, device, data.num_nodes)
            pbar.set_description(f'loss = {loss:3.4e}')
            history.append(loss)
            scheduler.step()

    logger.info('Finished training')
    model = model.to(torch.device("cpu"))

    logger.info('Running forward for all cells in CPU mode')
    embedded_cells = model.full_forward(x.cpu(), edge_index.cpu()).detach().numpy()

    outfbase = f'{ARGS.outprefix}-'+\
               f'{ARGS.hidden_features}hidden-'+\
               f'{ARGS.output_features}output-'+\
               f'{ARGS.num_layers}layer-'+\
               f'{ARGS.sampling}sample-'+\
               f'{ARGS.epochs}epoch'

    logger.info(f'saving to: {ARGS.outdir}/{outfbase}')

    np.save(f'{ARGS.outdir}/{outfbase}-embedding.npy', embedded_cells)
    np.save(f'{ARGS.outdir}/{outfbase}-barcodes.npy', barcodes)
    torch.save(model, f'{ARGS.outdir}/{outfbase}-model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', 
        help = "CSV with training data. 1 row per cell, columns: ['cell_id','x','y','label']."
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--maxdist', type=int, default=200)
    parser.add_argument('-o', '--outdir', type=str, default='./out')
    parser.add_argument('--outprefix', type=str, default='districts')
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--log', type=str, default='log.txt')

    parser.add_argument('--hidden_features', type=int, default=128)
    parser.add_argument('--output_features', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--sampling', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=3096)

    parser.add_argument('--lr_step', type=int, default=25)
    parser.add_argument('--lr_gamma', type=int, default=0.1)

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

    logger.info('Starting districts optimization')
    logger.info('ARGUMENTS:')
    for k,v in ARGS.__dict__.items():
        logger.info(f'\t{k}: {v}')

    main(ARGS, logger)