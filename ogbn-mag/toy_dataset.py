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

pyg_dataset = PygNodePropPredDataset(name='ogbn-mag', root=data_dir)
print(f"Intialization end... {time.time()-t0:.5f}")

split_index = pyg_dataset.get_idx_split()
print(f"split_index.keys() : {split_index.keys()}")
print(f"split_index['train'] : {split_index['train']}")
print(f"split_index['valid'] : {split_index['valid']}") 
print(f"split_index['test'] : {split_index['test']}")
print(f"split_index['train'].keys() : {split_index['train'].keys()}") # torch.int64

print()
print(f"split_index['train']['paper'].shape : {split_index['train']['paper'].shape}") # 629571
print(f"split_index['valid']['paper'].shape : {split_index['valid']['paper'].shape}") # 64879
print(f"split_index['test']['paper'].shape : {split_index['test']['paper'].shape}") # 41939
print(f"split_index['train']['paper'].dtype : {split_index['train']['paper'].dtype}") # torch.int64

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
print(f"data : {graph}")


#========================================================
# More detailed property
print(f"split_index['train'][:10] : {split_index['train']['paper'][:10]}")
print(f"split_index['valid'][:10] : {split_index['valid']['paper'][:10]}")
print(f"split_index['test'][:10] : {split_index['test']['paper'][:10]}")

print(f"graph.x_dict.keys() : {graph.x_dict.keys()}")
print(f"graph.x_dict['paper'].shape : {graph.x_dict['paper'].shape}")
print(f"graph.node_year : {graph.node_year}")
# print(f"graph.edge_reltype : {graph.edge_reltype}")
print(f"graph.edge_reltype.keys() : {graph.edge_reltype.keys()}")
print(f"graph.y_dict : {graph.y_dict}")

#print(f"graph.edge_index_dict : {graph.edge_index_dict}")
print(f"graph.edge_index_dict.keys() : {graph.edge_index_dict.keys()}")
print()
print(f"graph.edge_index_dict[('author', 'affiliated_with', 'institution')] : {graph.edge_index_dict[('author', 'affiliated_with', 'institution')]}")
print(f"graph.edge_index_dict[('author', 'writes', 'paper')] : {graph.edge_index_dict[('author', 'writes', 'paper')]}")
print(f"graph.edge_index_dict[('paper', 'cites', 'paper')] : {graph.edge_index_dict[('paper', 'cites', 'paper')]}")
print(f"graph.edge_index_dict[('paper', 'has_topic', 'field_of_study')] : {graph.edge_index_dict[('paper', 'has_topic', 'field_of_study')]}")
print()
print(f"graph.edge_index_dict[('author', 'affiliated_with', 'institution')].shape : {graph.edge_index_dict[('author', 'affiliated_with', 'institution')].shape}")
print(f"graph.edge_index_dict[('author', 'writes', 'paper')].shape : {graph.edge_index_dict[('author', 'writes', 'paper')].shape}")
print(f"graph.edge_index_dict[('paper', 'cites', 'paper')].shape : {graph.edge_index_dict[('paper', 'cites', 'paper')].shape}")
print(f"graph.edge_index_dict[('paper', 'has_topic', 'field_of_study')].shape : {graph.edge_index_dict[('paper', 'has_topic', 'field_of_study')].shape}")
print()
#print(f"graph.edge_feat_dict : {graph.edge_feat_dict}")
#print(f"graph.node_feat_dict : {graph.node_feat_dict}")
print(f"graph.num_nodes_dict : {graph.num_nodes_dict}")
print(f"graph.num_nodes_dict.keys() : {graph.num_nodes_dict.keys()}")


# Test node also have label...? What..?

#print(graph.edge_index('paper','paper')) # impossible code


