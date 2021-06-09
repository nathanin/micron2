import numpy as np
import networkx as nx
import pandas as pd

import itertools
import tqdm.auto as tqdm
from scipy.spatial import Delaunay

"""
Spatial interaction utilities 

Possible functions:

  is_interacting(cellA, cellB, graph)
    check if cellA and cellB are connected in the graph
  
  interaction_table(coords, labels, query, target, graph_type='Delaunay')
    produce a table of cell-level interaction information for spatial associations 
      between the `query` and `target` celltypes 

    for each query cell produce a table:
      cell_id, is_interacting, partner_celltype, partner_ids, n_partners

  interaction_test(coords, labels, query, target, graph_type='Delaunay')
    permutation test for interaction frequency between `query` and `target` cells

  delaunay_graphlets(coords, labels)
    produce a histogram of celltype graphlets
      a cell type graphlet can be defined several ways, the way we think about here
      is a triangle with labelled vertices, and the 'graphlet' will be the cell
      identities that are on the triangle

  interactions(coords, labels, graph_type='Delaunay')
    possible wrapper function to run multiple rounds of query/target interaction tests

"""

def point_dist(points, p1, p2):
  x1,y1 = points[p1]
  x2,y2 = points[p2]
  return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def tri2edges(tri, coords, max_dist=50):
  edge_index_list = []
  for s in tri.simplices:
    for p1,p2 in itertools.combinations(s, 2):
      d = point_dist(coords, p1,p2)
      if d > max_dist: continue
      edge_index_list.append([p1,p2])
      edge_index_list.append([p2,p1]) #add also the reverse edge; we're undirected.
  return edge_index_list

def coords2neighbors(coords, max_dist=50):
  tri = Delaunay(coords)
  edges = tri2edges(tri, coords, max_dist=max_dist)
  G = nx.Graph()
  G.add_nodes_from(np.arange(coords.shape[0]))
  G.add_edges_from(edges)
  adj = nx.adjacency_matrix(G)
  return adj