# SessionCQA
The repository holds the code for mining logical queries/creating user intention KG from session data. 


## Data

Just download the data by clicking this [**link**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jbai_connect_ust_hk/EmdRTPX0S_9EgbXsPJ_hgmYBZIPaJhxz59XKOfs1PQTJ-A?e=AMEcaj), and extract them in root of this repository. Three datasets are all included. There is more pre-processing for LSGT and some of the baselines, so please remember that using the correct corresponding files otherwise will cause data-loading errors. 
It is too abstract to describe how to run the code because there are many combinations of logical encoder, session encoder, and datasets. We make it clearer by the following example scripts, and you can substitute the flags.


## Baselines of Logical Encoder + Session Encoder


You can run the following script for logical encoder + session encoder on the Amazon dataset

```
python ./deduction_model/train.py \
-dn hyper_graph_data_en \
-m fuzzqe \
--train_query_dir ./sampled_hyper_train_merged \
--valid_query_dir ./sampled_hyper_valid \
--test_query_dir ./sampled_hyper_test \
--checkpoint_path ./logs \
-fol  \
-b 512 \
--log_steps 120000 \
-lr 0.001 \
--session_encoder AttnMixer
```

The ```-m fuzzqe``` denotes the logical encoder which we use here. The ```--session_encoder''' denotes the session encoder here. In this  implementation, we implemented the combinations of these two flags, which have the following options.


### Baseline Logical Encoders

The baseline logical encoder implementations are adopted from the SQE repo, a general code base. 
Here is the code for the 

| Model Flag (-m) | Paper  |
|---|---|
| gqe |  [Embedding logical queries on knowledge graphs](https://proceedings.neurips.cc/paper/2018/hash/ef50c335cca9f340bde656363ebd02fd-Abstract.html)  |
| q2b | [Query2box: Reasoning over knowledge graphs in vector space using box embeddings](https://openreview.net/forum?id=BJgr4kSFDS) |
| betae | [Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs](https://proceedings.neurips.cc/paper/2020/hash/e43739bba7cdb577e9e3e4e42447f5a5-Abstract.html)  |
| cone | [Cone: Cone embeddings for multihop reasoning over knowledge graphs](https://openreview.net/pdf?id=Twf_XYunk5j) |
| q2p |  [Query2Particles: Knowledge Graph Reasoning with Particle Embeddings](https://aclanthology.org/2022.findings-naacl.207/) |
| fuzzqe | [Fuzzy Logic Based Logical Query Answering on Knowledge Graphs](https://arxiv.org/abs/2108.02390) |
| tree_lstm |[Sequential Query Encoding for Complex Query Answering on Knowledge Graphs](https://openreview.net/pdf?id=ERqGqZzSu5) |


### Session Encode


Here are the options for the session encoder flags. 


| Session Encoder Flag (--session_encoder) | Paper  |
|---|---|
| GRURec | [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555)  |
| SRGNN | [Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855) |
| AttnMixer | [Efficiently Leveraging Multi-level User Intent for Session-based Recommendation via Atten-Mixer Network](https://dl.acm.org/doi/10.1145/3539597.3570445)|



## Baselines of NQE and SQE on Dressipi
```
python ./deduction_model/train.py \
-dn hyper_graph_dressipi \
-m nqe \
--train_query_dir ./sampled_hyper_train_dressipi_merged \
--valid_query_dir ./sampled_hyper_valid_dressipi \
--test_query_dir ./sampled_hyper_test_dressipi \
--checkpoint_path ./logs \
-fol  \
-b 512 \
--log_steps 120000 \
-lr 0.0001
```


## Running LSGT on Diginetica
Note that there is a trailing ```_dgl``` in the queries dirs, in which we use dgl to convert the query graph into graph tokens. 

```
python ./deduction_model/train.py \
-dn hyper_graph_diginetica \
-m lsgt \
--train_query_dir ./sampled_hyper_train_diginetica_merged_graph_dgl \
--valid_query_dir ./sampled_hyper_valid_dininetica_graph_dgl \
--test_query_dir ./sampled_hyper_test_diginetica_graph_dgl \
--checkpoint_path ./logs \
--num_layers 2 \
-fol  \
-b 1024 \
--log_steps 120000 \
-lr 0.001

```

## Scripts to reproduce the result in this paper

We also include the shell scripts that we run to reproduce the number shown in the paper. 


## Others

If anything is unclear, please get in touch with Jiaxin Bai via jbai@connect.ust.hk.
If you find this useful, please cite:

```
@article{DBLP:journals/corr/abs-2312-13866,
  author       = {Jiaxin Bai and
                  Chen Luo and
                  Zheng Li and
                  Qingyu Yin and
                  Yangqiu Song},
  title        = {Understanding Inter-Session Intentions via Complex Logical Reasoning},
  journal      = {CoRR},
  volume       = {abs/2312.13866},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2312.13866},
  doi          = {10.48550/ARXIV.2312.13866},
  eprinttype    = {arXiv},
  eprint       = {2312.13866},
  timestamp    = {Wed, 17 Jan 2024 17:14:38 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2312-13866.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```





