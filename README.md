# Sequence to General Tree: Knowledge-Guided Geometry Word Problem Solving

This is the official repository for the ACL 2021 paper "Sequence to General Tree: Knowledge-Guided Geometry Word Problem Solving". In this paper, we introduce a **sequence-to-general-tree** architecture which aims to generalize previous sequence-to-tree models on the Math Word Problem Solving task. 
folr the ACL 2021 paper


where we introduce a 



## Code and Data



## Prerequisites

This project is written based on `AllenNLP` and `Pytorch Geometric` frameworks.

1. If you only want to run **S2G**, please install `AllenNLP` as follows:

```
pip install allennlp
```

2. If you want to run **S2G+KG**, please follow the instructions to install `Pytorch Geometric`  
https://github.com/rusty1s/pytorch_geometric



## Run S2G / S2G+KG

We provide

+ ``: run the configs in ....

```
bash run_.sh RESULT_DIR
```




## Evaluate

!python evaluate.py results/s2g-tune/check_loss/ --five_fold


