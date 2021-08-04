# Sequence to General Tree: Knowledge-Guided Geometry Word Problem Solving

This is the official repository for the ACL 2021 paper [Sequence to General Tree: Knowledge-Guided Geometry Word Problem Solving](https://aclanthology.org/2021.acl-short.121/). In this paper, we introduce a **sequence-to-general-tree (S2G)** architecture which aims to generalize the previous sequence-to-tree models. Instead of mapping the problem text into a binary expression tree, our S2G can learn to map the problem text into an operation tree where the nodes can be formulas with arbitrary number of children (from unary to N-ary). In this way, we can incorporate domain knowledge (formulas) into problem-solving, making the results more neat and interpretable as well as improving the solver performance. As illustrated below, S2G can output operation trees (figure(b)) that explain the formulas used in solving problems.

<br>
<p align="center">
  <img src="./imgs/figure1.png" width="550">
</p>


## Code and Data

We implement S2G using Pytorch and AllenNLP, and this repository is organized following the standard AllenNLP setting.
Here, we give a brief description of each folder/file and their content:

+ `data`: 
    + `geometryQA`: the GeometryQA dataset.
    + `geometryQA_5fold`: the subsets of GeometryQA for 5-fold cross validation.
+ `libs`: the source code for `dataset_reader`, `models`, `modules`, and `tools`.
+ `configs`: the configuration files for different experiment settings.  
+ `run_xxx.sh`: the scripts to run the experiment settings defined in `configs`.


## Prerequisites

This project is written based on `AllenNLP` and `Pytorch Geometric` frameworks.

1. If you only want to run **S2G**, please install `AllenNLP` as follows:

```
pip install allennlp
```

2. If you want to run **S2G+KG**, please follow the instructions to install `Pytorch Geometric`  
https://github.com/rusty1s/pytorch_geometric



## Run S2G / S2G+KG

Here are the shell scripts to run the experiments.

+ `run_s2g.sh`: run **S2G** on GeometryQA.
+ `run_s2gkg.sh`: run **S2G+KG** on GeometryQA.

We also provide scripts to run the experiments using AllenNLP `bucket_sampler`.

+ `run_s2g_bucket.sh`: run **S2G** on GeometryQA using bucket_sampler.
+ `run_s2gkg_bucket.sh`: run **S2G+KG** on GeometryQA using bucket_sampler.


To run the shell scripts above, please use the following command and specify the location for the `RESULT_DIR`. The command will create a `results/RESULT_DIR` folder to store the model parameters and experimental results. 
```
bash run_XXX.sh RESULT_DIR
```

Using the following command to evaluate the performance.
```
python evaluate.py results/RESULT_DIR --five_fold
```

## Cite us

You can cite us using the following BibTeX:

```
@inproceedings{tsai-etal-2021-sequence,
    title = "Sequence to General Tree: Knowledge-Guided Geometry Word Problem Solving",
    author = "Tsai, Shih-hung and Liang, Chao-Chun and Wang, Hsin-Min and Su, Keh-Yih",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.121",
    doi = "10.18653/v1/2021.acl-short.121",
    pages = "964--972",
}
```
