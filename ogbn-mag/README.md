# ogbn-mag

### Requirements

### Data preprocessing
```
python ogbn-mag/ogb_custum/nodeproppred/dataset_pyg.py
python ogbn-mag/data_preprocess.py
```

### Data preprocessing for sampler
```
python ogbn-mag/china_preprocess.py
python ogbn-mag/toggle_preprocess.py
```

### Train baseling
```
python ogbn-mag/baseline_acua.py --link=toggle --cross_partition_number=1
```