import argparse
import glob
import os
import os.path as osp
import sys
import time
from typing import List, NamedTuple, Optional
sys.path.insert(0,'~/Adaptive_Sampling')    
import numpy as np
import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
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

data_dir="/users/PAS1289/oiocha/Adaptive_Sampling/dataset"
pyg_dataset = PygNodePropPredDataset(name='ogbn-papers100M', root=data_dir)
dataset=pyg_dataset[0]
print(f"type(dataset) : {type(dataset)}")

data_dir="/users/PAS1289/oiocha/Adaptive_Sampling/dataset/ogbn_papers100M"
path = f'{data_dir}/full_adj_t.pt'
if not osp.exists(path):  # Will take approximately 16 minutes...
    t = time.perf_counter()
    print('Merging adjacency matrices...', end=' ', flush=True)

    edge_index = dataset.edge_index('paper', 'paper')
    row, col = torch.from_numpy(edge_index)
    rows = [row, col]
    cols = [col, row]

    edge_index = dataset.edge_index('author', 'writes', 'paper')
    row, col = torch.from_numpy(edge_index)
    row += dataset.num_papers
    rows += [row, col]
    cols += [col, row]

    edge_index = dataset.edge_index('author', 'institution')
    row, col = torch.from_numpy(edge_index)
    row += dataset.num_papers
    col += dataset.num_papers + dataset.num_authors
    rows += [row, col]
    cols += [col, row]

    edge_types = [
        torch.full(x.size(), i, dtype=torch.int8)
        for i, x in enumerate(rows)
    ]

    row = torch.cat(rows, dim=0)
    del rows
    col = torch.cat(cols, dim=0)
    del cols

    N = (dataset.num_papers + dataset.num_authors +
            dataset.num_institutions)

    perm = (N * row).add_(col).numpy().argsort()
    perm = torch.from_numpy(perm)
    row = row[perm]
    col = col[perm]

    edge_type = torch.cat(edge_types, dim=0)[perm]
    del edge_types

    full_adj_t = SparseTensor(row=row, col=col, value=edge_type,
                                sparse_sizes=(N, N), is_sorted=True)

    torch.save(full_adj_t, path)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

#path = f'{dataset.dir}/full_feat.npy'
#done_flag_path = f'{dataset.dir}/full_feat_done.txt'
#log_path = f'{dataset.dir}/rgnn_log.txt'
NEWROOT='/fs/scratch/PAS1289/data'
path = NEWROOT+'/full_feat.npy'
done_flag_path = NEWROOT+'/full_feat_done.txt'
log_path = NEWROOT+'/rgnn_log.txt'

if not osp.exists(done_flag_path):  # Will take ~3 hours...
    t = time.perf_counter()
    fl=open(log_path,'w')

    print('Generating full feature matrix...')
    fl.write('Generating full feature matrix...')
    fl.write('\n')
    fl.flush()
    node_chunk_size = 100000
    dim_chunk_size = 64
    N = (dataset.num_papers + dataset.num_authors +
            dataset.num_institutions)

    paper_feat = dataset.paper_feat
    x = np.memmap(path, dtype=np.float16, mode='w+',
                    shape=(N, self.num_features))

    t0=time.time()
    print('Copying paper features...','commit -m 1010 UPD')
    fl.write('Copying paper features...')
    fl.write('\n')
    fl.flush()
    for i in range(0, dataset.num_papers, node_chunk_size):
        if ((i/node_chunk_size)%10==0):
            print("COPY - Progress... :",i,"/",dataset.num_papers,"Consumed time :",time.time()-t0)
            fl.write("COPY - Progress... :"+str(i)+"/"+str(dataset.num_papers)+"| Consumed time :"+str(time.time()-t0))
            fl.write('\n')
            fl.flush()
        j = min(i + node_chunk_size, dataset.num_papers)
        x[i:j] = paper_feat[i:j]
    print("h1")
    edge_index = dataset.edge_index('author', 'writes', 'paper')
    print("h2")
    row, col = torch.from_numpy(edge_index)
    print("h3")
    adj_t = SparseTensor(
        row=row, col=col,
        sparse_sizes=(dataset.num_authors, dataset.num_papers),
        is_sorted=True)
    print("h4")
    # Processing 64-dim subfeatures at a time for memory efficiency.
    print('Generating author features...')
    fl.write('Generating author features...')
    fl.write('\n')
    fl.flush()
    t0=time.time()
    for i in range(0, self.num_features, dim_chunk_size):
        print("GEN_author Progress... ",i,"/",self.num_features/dim_chunk_size,"Consumed time :",time.time()-t0)
        fl.write("GEN_author Progress... "+str(i)+"/"+str(self.num_features/dim_chunk_size)+"| Consumed time :"+str(time.time()-t0))
        fl.write('\n')
        fl.flush()
        j = min(i + dim_chunk_size, self.num_features)
        inputs = get_col_slice(fl, paper_feat, start_row_idx=0,
                                end_row_idx=dataset.num_papers,
                                start_col_idx=i, end_col_idx=j)
        inputs = torch.from_numpy(inputs)
        outputs = adj_t.matmul(inputs, reduce='mean').numpy()
        del inputs
        save_col_slice(
            fl, x_src=outputs, x_dst=x, start_row_idx=dataset.num_papers,
            end_row_idx=dataset.num_papers + dataset.num_authors,
            start_col_idx=i, end_col_idx=j)
        del outputs
    print("h5")
    edge_index = dataset.edge_index('author', 'institution')
    row, col = torch.from_numpy(edge_index)
    adj_t = SparseTensor(
        row=col, col=row,
        sparse_sizes=(dataset.num_institutions, dataset.num_authors),
        is_sorted=False)
    
    print('Generating institution features...')
    fl.write('Generating institution features...')
    fl.write('\n')
    fl.flush()
    t0=time.time()
    # Processing 64-dim subfeatures at a time for memory efficiency.
    for i in range(0, self.num_features, dim_chunk_size):
        print("GEN_IN Progress... ",i,"/",self.num_features/dim_chunk_size,"Consumed time :",time.time()-t0)
        fl.write("GEN_IN Progress... "+str(i)+"/"+str(self.num_features/dim_chunk_size)+"| Consumed time :"+str(time.time()-t0))
        fl.write('\n')
        fl.flush()
        j = min(i + dim_chunk_size, self.num_features)
        inputs = get_col_slice(
            fl, x, start_row_idx=dataset.num_papers,
            end_row_idx=dataset.num_papers + dataset.num_authors,
            start_col_idx=i, end_col_idx=j)
        inputs = torch.from_numpy(inputs)
        outputs = adj_t.matmul(inputs, reduce='mean').numpy()
        del inputs
        save_col_slice(
            fl, x_src=outputs, x_dst=x,
            start_row_idx=dataset.num_papers + dataset.num_authors,
            end_row_idx=N, start_col_idx=i, end_col_idx=j)
        del outputs
    print("h6")
    x.flush()
    del x
    print(f'Done! [{time.perf_counter() - t:.2f}s]')

    with open(done_flag_path, 'w') as f:
        f.write('done')
    fl.close()
path = f'{dataset.dir}/full_feat.npy'