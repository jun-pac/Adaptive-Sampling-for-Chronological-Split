
import argparse
import glob
import os
import os.path as osp
import sys
import time
from typing import List, NamedTuple, Optional
sys.path.insert(0,'~/OGB-NeurIPS-Team-Park')    
import numpy as np
import torch
import torch.nn.functional as F
from ogb_custum.nodeproppred import PygNodePropPredDataset
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import Accuracy
from torch import Tensor
from torch.nn import BatchNorm1d, Dropout, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import GATConv, SAGEConv
from torch_sparse import SparseTensor
from tqdm import tqdm
from multiprocessing import Pool, Process, Array

t0=time.time()

data_dir="~/Adaptive_Sampling/dataset"
dataset_name='ogbn-mag'
num_features=128
pyg_dataset = PygNodePropPredDataset(name=dataset_name, root=data_dir)
dataset=pyg_dataset[0]

dataset_name = '_'.join(dataset_name.split('-')) 
data_dir=os.path.join(data_dir,dataset_name)
path = f'{data_dir}/full_adj_t.pt'

num_papers=dataset.num_nodes_dict['paper']
num_authors=dataset.num_nodes_dict['author']
num_institutions=dataset.num_nodes_dict['institution']
num_fields=dataset.num_nodes_dict['field_of_study']
N = (num_papers + num_authors + num_institutions + num_fields)
nums=[num_papers, num_papers + num_authors, num_papers + num_authors + num_institutions, N]

print("Loading data...")
adj_t = torch.load(path)
rowptr,col,_=adj_t.csr()
print(f"Done! {time.time()-t0}")

relation_ptr=torch.Tensor(4*N+1).to(torch.long)


print(f"col length : {len(col)}")
print(f"rowptr length : {len(rowptr)}")
print(f"N : {N}")

relation_ptr.share_memory_()
relation_ptr[0]=0
print(f"[0] : {relation_ptr[0].dtype}")
print(f"rowptr_dtype : {rowptr[0].dtype}")

def task(i):
    row_sz=rowptr[i+1]-rowptr[i]
    j=0
    while (j<row_sz and col[rowptr[i]+j]<nums[0]):
        j+=1
    relation_ptr[4*i+1]=rowptr[i]+j
    
    while (j<row_sz and col[rowptr[i]+j]<nums[1]):
        j+=1
    relation_ptr[4*i+2]=rowptr[i]+j

    while (j<row_sz and col[rowptr[i]+j]<nums[2]):
        j+=1
    relation_ptr[4*i+3]=rowptr[i]+j

    relation_ptr[4*i+4]=rowptr[i+1]

    if i%1000000==0:
        print(f"{i}th row : {relation_ptr[4*i+1]}, {relation_ptr[4*i+2]}, {relation_ptr[4*i+3]}, {relation_ptr[4*i+4]} - time : {time.time()-t0:.2f}")

num = 48
pool = Pool(num)
pool.map(task, range(N)) # This would take 2120s

print(f"first ten relation_ptr(before save) : {relation_ptr[:10]}")

torch.save(relation_ptr, "~/Adaptive_Sampling/ogbn-mag/sampler/bi_relation_ptr.pt")