# Pointcloud Representation Learning with GNN

```shell
python src/train.py
```

with or without grasping data can be set in `configs/data/oakink.yaml`

set `grasp` to `True` to use grasping data

set `grasp` to `False` to use only object pointcloud

run `eval.py` to generate the embedding of the whole dataset

```shell
python src/eval.py
```

run `cluster.py` to generate the K-means clustering result

```shell
python scripts/cluster.py
```
