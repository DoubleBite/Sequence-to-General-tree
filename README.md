# Sequence to General Tree: Knowledge-Guided Geometry Word Problem Solving

This repo
ACL paper





## Prerequisites

This project is written based on `AllenNLP` and `Pytorch Geometric` frameworks.

1. If you only want to try S2G, please install `AllenNLP` as follows:

```
pip install allennlp
```

2. If you want to try S2G+KG please follow the tutorial in the link below to install `Pytorch Geometric`  
https://github.com/rusty1s/pytorch_geometric



## Run S2G / S2G+KG / Baselines

We provide

+ ``: run the configs in ....

```
bash run_.sh RESULT_DIR
```




## Evaluate

!python evaluate.py results/s2g-tune/check_loss/ --five_fold


