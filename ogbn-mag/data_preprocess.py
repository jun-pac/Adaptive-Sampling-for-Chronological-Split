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

data_dir="/users/PAS1289/oiocha/Adaptive_Sampling/dataset"
dataset_name='ogbn-mag'
num_features=128
pyg_dataset = PygNodePropPredDataset(name=dataset_name, root=data_dir)
dataset=pyg_dataset[0]

dataset_name = '_'.join(dataset_name.split('-')) 
data_dir=os.path.join(data_dir,dataset_name)
path = f'{data_dir}/full_adj_t.pt'
path_mono = f'{data_dir}/mono_adj_t.pt'

num_papers=dataset.num_nodes_dict['paper']
num_authors=dataset.num_nodes_dict['author']
num_institutions=dataset.num_nodes_dict['institution']
num_fields=dataset.num_nodes_dict['field_of_study']
N = num_papers + num_authors + num_institutions + num_fields


# This will take approximately 5 seconds...
if not osp.exists(path):
    t = time.perf_counter()
    print('Merging adjacency matrices...', end=' ', flush=True)

    row, col = dataset.edge_index_dict[('paper', 'cites', 'paper')]
    row, col = torch.clone(row), torch.clone(col)
    rows = [row, col]
    cols = [col, row]

    row, col = dataset.edge_index_dict[('author', 'writes', 'paper')]
    row, col = torch.clone(row), torch.clone(col)
    row += num_papers
    rows += [row, col]
    cols += [col, row]

    row, col = dataset.edge_index_dict[('author', 'affiliated_with', 'institution')]
    row, col = torch.clone(row), torch.clone(col)
    row += num_papers
    col += num_papers + num_authors
    rows += [row, col]
    cols += [col, row]

    row, col = dataset.edge_index_dict[('paper', 'has_topic', 'field_of_study')]
    row, col = torch.clone(row), torch.clone(col)
    col += num_papers + num_authors + num_institutions
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

'''
max(row) : 736388
min(row) : 0
max(row) : 1134648
min(row) : 0
max(col) : 8739
min(col) : 0
max(col) : 59964
min(col) : 0
'''


if not osp.exists(path_mono):
    t = time.perf_counter()
    print('Merging adjacency matrices...', end=' ', flush=True)

    row, col = dataset.edge_index_dict[('paper', 'cites', 'paper')]
    row, col = torch.clone(row), torch.clone(col)
    rows = [row]
    cols = [col]

    row, col = dataset.edge_index_dict[('author', 'writes', 'paper')]
    row, col = torch.clone(row), torch.clone(col)
    row += num_papers
    rows += [row, col]
    cols += [col, row]

    row, col = dataset.edge_index_dict[('author', 'affiliated_with', 'institution')]
    row, col = torch.clone(row), torch.clone(col)
    row += num_papers
    col += num_papers + num_authors
    rows += [row, col]
    cols += [col, row]

    row, col = dataset.edge_index_dict[('paper', 'has_topic', 'field_of_study')]
    row, col = torch.clone(row), torch.clone(col)
    col += num_papers + num_authors + num_institutions
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

    perm = (N * row).add_(col).numpy().argsort()
    perm = torch.from_numpy(perm)
    row = row[perm]
    col = col[perm]

    edge_type = torch.cat(edge_types, dim=0)[perm]
    del edge_types

    mono_adj_t = SparseTensor(row=row, col=col, value=edge_type,
                                sparse_sizes=(N, N), is_sorted=True)

    torch.save(mono_adj_t, path_mono)
    print(f'Done! [{time.perf_counter() - t:.2f}s]')


    
NEWROOT=data_dir
path = NEWROOT+'/full_feat.npy'
done_flag_path = NEWROOT+'/full_feat_done.txt'
log_path = NEWROOT+'/rgnn_log.txt'


def get_col_slice(fl, x, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    outs = []
    chunk = 100000
    print("get_col_slice")
    fl.write("get_col_slice")
    fl.write('\n')
    fl.flush()
    t0=time.time()
    for i in range(start_row_idx, end_row_idx, chunk):
        j = min(i + chunk, end_row_idx)
        outs.append(x[i:j, start_col_idx:end_col_idx].copy())
    return np.concatenate(outs, axis=0)

def save_col_slice(fl, x_src, x_dst, start_row_idx, end_row_idx, start_col_idx,
                   end_col_idx):
    assert x_src.shape[0] == end_row_idx - start_row_idx
    assert x_src.shape[1] == end_col_idx - start_col_idx
    chunk, offset = 100000, start_row_idx
    print("save_col_slice")
    fl.write("save_col_slice")
    fl.write('\n')
    fl.flush()
    t0=time.time()
    for i in range(0, end_row_idx - start_row_idx, chunk):
        if((i/chunk)%10==0):
            print("SAVE - Sub routine...",i, "/",(end_row_idx - start_row_idx),"| time :",time.time()-t0)
            fl.write("SAVE - Sub routine..."+str(i)+"/"+str(end_row_idx - start_row_idx)+"| time :"+str(time.time()-t0))
            fl.write('\n')
            fl.flush()
        j = min(i + chunk, end_row_idx - start_row_idx)
        x_dst[offset + i:offset + j, start_col_idx:end_col_idx] = x_src[i:j]

if not osp.exists(done_flag_path):  # Will take ~3 hours...
    t = time.perf_counter()
    fl=open(log_path,'w')

    print('Generating full feature matrix...')
    fl.write('Generating full feature matrix...')
    fl.write('\n')
    fl.flush()
    node_chunk_size = 100000
    dim_chunk_size = 64

    paper_feat = dataset.x_dict['paper'].numpy()
    x = np.memmap(path, dtype=np.float16, mode='w+', shape=(N, num_features))

    t0=time.time()
    print('Copying paper features...','commit -m 1010 UPD')
    fl.write('Copying paper features...')
    fl.write('\n')
    fl.flush()
    for i in range(0, num_papers, node_chunk_size):
        if ((i/node_chunk_size)%10==0):
            print("COPY - Progress... :",i,"/",num_papers,"Consumed time :",time.time()-t0)
            fl.write("COPY - Progress... :"+str(i)+"/"+str(num_papers)+"| Consumed time :"+str(time.time()-t0))
            fl.write('\n')
            fl.flush()
        j = min(i + node_chunk_size, num_papers)
        x[i:j] = paper_feat[i:j]


    row, col = dataset.edge_index_dict[('author', 'writes', 'paper')]
    row, col = torch.clone(row), torch.clone(col)
    #row, col = torch.from_numpy(edge_index)
    #print(f"Author - Paper : {max(row)} - {max(col)}") # Author - Paper : 1871037 - 736388
    # It was deep copy issue.

    adj_t = SparseTensor(
        row=row, col=col,
        sparse_sizes=(num_authors, num_papers),
        is_sorted=True)
    # Processing 64-dim subfeatures at a time for memory efficiency.
    print('Generating author features...')
    fl.write('Generating author features...')
    fl.write('\n')
    fl.flush()
    t0=time.time()
    for i in range(0, num_features, dim_chunk_size):
        print("GEN_author Progress... ",i,"/",num_features/dim_chunk_size,"Consumed time :",time.time()-t0)
        fl.write("GEN_author Progress... "+str(i)+"/"+str(num_features/dim_chunk_size)+"| Consumed time :"+str(time.time()-t0))
        fl.write('\n')
        fl.flush()
        j = min(i + dim_chunk_size, num_features)
        inputs = get_col_slice(fl, paper_feat, start_row_idx=0,
                                end_row_idx=num_papers,
                                start_col_idx=i, end_col_idx=j)
        inputs = torch.from_numpy(inputs)
        outputs = adj_t.matmul(inputs, reduce='mean').numpy()
        del inputs
        save_col_slice(
            fl, x_src=outputs, x_dst=x, start_row_idx=num_papers,
            end_row_idx=num_papers + num_authors,
            start_col_idx=i, end_col_idx=j)
        del outputs

    

    col, row = dataset.edge_index_dict[('paper', 'has_topic', 'field_of_study')]
    row, col = torch.clone(row), torch.clone(col)
    #col, row = torch.from_numpy(edge_index)
    #print(f"Field - Paper : {max(row)} - {max(col)}")
    adj_t = SparseTensor(
        row=row, col=col,
        sparse_sizes=(num_fields, num_papers),
        is_sorted=True)

    # Processing 64-dim subfeatures at a time for memory efficiency.
    print('Generating field features...')
    fl.write('Generating field features...')
    fl.write('\n')
    fl.flush()
    t0=time.time()
    for i in range(0, num_features, dim_chunk_size):
        print("GEN_field Progress... ",i,"/",num_features/dim_chunk_size,"Consumed time :",time.time()-t0)
        fl.write("GEN_field Progress... "+str(i)+"/"+str(num_features/dim_chunk_size)+"| Consumed time :"+str(time.time()-t0))
        fl.write('\n')
        fl.flush()
        j = min(i + dim_chunk_size, num_features)
        inputs = get_col_slice(fl, paper_feat, start_row_idx=0,
                                end_row_idx=num_papers,
                                start_col_idx=i, end_col_idx=j)
        inputs = torch.from_numpy(inputs)
        outputs = adj_t.matmul(inputs, reduce='mean').numpy()
        del inputs
        save_col_slice(
            fl, x_src=outputs, x_dst=x, start_row_idx=num_papers+num_authors+num_institutions,
            end_row_idx=num_papers+num_authors+num_institutions+num_fields,
            start_col_idx=i, end_col_idx=j)
        del outputs



    row, col = dataset.edge_index_dict[('author', 'affiliated_with', 'institution')]
    row, col = torch.clone(row), torch.clone(col)
    #row, col = torch.from_numpy(edge_index)
    #print(f"Author - Institiution : {max(row)} - {max(col)}") # Author - Institiution : 1134648 - 8739
    adj_t = SparseTensor(
        row=col, col=row,
        sparse_sizes=(num_institutions, num_authors),
        is_sorted=False)
    print('Generating institution features...')
    fl.write('Generating institution features...')
    fl.write('\n')
    fl.flush()
    t0=time.time()
    # Processing 64-dim subfeatures at a time for memory efficiency.
    for i in range(0, num_features, dim_chunk_size):
        print("GEN_IN Progress... ",i,"/",num_features/dim_chunk_size,"Consumed time :",time.time()-t0)
        fl.write("GEN_IN Progress... "+str(i)+"/"+str(num_features/dim_chunk_size)+"| Consumed time :"+str(time.time()-t0))
        fl.write('\n')
        fl.flush()
        j = min(i + dim_chunk_size, num_features)
        inputs = get_col_slice(
            fl, x, start_row_idx=num_papers,
            end_row_idx=num_papers + num_authors,
            start_col_idx=i, end_col_idx=j)
        inputs = torch.from_numpy(inputs)
        outputs = adj_t.matmul(inputs, reduce='mean').numpy()
        del inputs
        save_col_slice(
            fl, x_src=outputs, x_dst=x,
            start_row_idx=num_papers + num_authors,
            end_row_idx=num_papers + num_authors + num_institutions, start_col_idx=i, end_col_idx=j)
        del outputs
    x.flush()
    del x
    print(f'Done! [{time.perf_counter() - t:.2f}s]')


    with open(done_flag_path, 'w') as f:
        f.write('done')
    fl.close()
