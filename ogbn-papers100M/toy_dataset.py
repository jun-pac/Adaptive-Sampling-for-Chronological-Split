import argparse
import glob
import os
import os.path as osp
import sys
import time
from typing import List, NamedTuple, Optional
sys.path.append('/users/PAS1289/oiocha/Adaptive_Sampling')    
import numpy as np
import torch
from ogb_custum.nodeproppred import PygNodePropPredDataset
from tqdm import tqdm
t0=time.time()

data_dir="/users/PAS1289/oiocha/Adaptive_Sampling/dataset"

pyg_dataset = PygNodePropPredDataset(name='ogbn-papers100M', root=data_dir)
print(f"Intialization end... {time.time()-t0:.5f}")

split_index = pyg_dataset.get_idx_split()
print(f"split_index.keys() : {split_index.keys()}")
print(f"split_index['train'].shape : {split_index['train'].shape}") # 1207179
print(f"split_index['valid'].shape : {split_index['valid'].shape}") # 125265
print(f"split_index['test'].shape : {split_index['test'].shape}") # 214338
print(f"split_index['train'].dtype : {split_index['train'].dtype}") # torch.int64

'''
gdp=torch.load("/users/PAS1289/oiocha/Adaptive_Sampling/dataset/ogbn_papers100M/processed/geometric_data_processed.pt")
print("geometric_data_processed ==== gdp")
print(f"gdp.shape : {gdp.shape}, time : {time.time()-t0:.5f}")
print(f"gdp.dtype : {gdp.dtype}")
'''

t0=time.time()
graph = pyg_dataset[0]
print(f"Graph data loading time : {time.time()-t0:.5f}")
print(f"type(data) : {type(graph)}")
print(f"data.x : {graph.x}")
print(f"data.edge_index : {graph.edge_index}")
print(f"data.x : {graph.num_nodes}")

#  data : Data(edge_index=[2, 1615685872], x=[111059956, 128], node_year=[111059956, 1], y=[111059956, 1]

#========================================================
# More detailed property
print(f"split_index['train'][:10] : {split_index['train'][:10]}")
print(f"split_index['valid'][:10] : {split_index['valid'][:10]}")
print(f"split_index['test'][:10] : {split_index['test'][:10]}")

for i in range(10):
    print(f"{i}th train node | x : {graph.x[split_index['train'][i]][:3]} | y : {graph.y[split_index['train'][i]].item()} | year : {graph.node_year[split_index['train'][i]].item()}")
print()
for i in range(10):
    print(f"{i}th valid node | x : {graph.x[split_index['valid'][i]][:3]} | y : {graph.y[split_index['valid'][i]].item()} | year : {graph.node_year[split_index['train'][i]].item()}")
print()
for i in range(10):
    print(f"{i}th test node | x : {graph.x[split_index['test'][i]][:3]} | y : {graph.y[split_index['test'][i]].item()} | year : {graph.node_year[split_index['train'][i]].item()}")
print()
for i in range(0,111059956, 25777777):
    print(f"{i}th random node | x : {graph.x[i][:3]} | y : {graph.y[i].item()} | year : {graph.node_year[i].item()}")
print()
print(f"graph.edge_index_dict : {graph.edge_index_dict}")
print(f"graph.edge_feat_dict : {graph.edge_feat_dict}")
print(f"graph.node_feat_dict : {graph.node_feat_dict}")
print(f"graph.num_nodes_dict : {graph.num_nodes_dict}")


# Test node also have label...? What..?

#print(graph.edge_index('paper','paper')) # impossible code





#  data : Data(edge_index=[2, 1615685872], x=[111059956, 128], node_year=[111059956, 1], y=[111059956, 1])