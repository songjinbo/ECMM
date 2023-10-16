# Rethinking Large-scale Pre-ranking System: Entire-chain Cross-domain Models

This repo is the official implementation for the [CIKM 2022 paper: *Rethinking Large-scale Pre-ranking System: Entire-chain Cross-domain Models*](https://dl.acm.org/doi/10.1145/3511808.3557683).

The paper is also available in [arxiv: *Rethinking Large-scale Pre-ranking System: Entire-chain Cross-domain Models*](https://arxiv.org/abs/2310.08039)

## Dataset

Our released dataset ***JD.ad Pre-ranking*** could be found at [JD JingPan](http://box.jd.com/sharedInfo/FC5D1899151CF81DEEC946AFBC5707A7) with password `gj33tv`.

## Introduction

* *src/* : code including model structure and metric calculating

* *configure/* : configuration corresponding to each model

* *model/* : echo folder of which include offline metric such as auc, gauc and recall_rate(*metric_res.txt*). It's generated after train procedure finish

* *data/* : used to save train and test dataset

## Requirements

* conda 4.6.14
* python 3.6.13
* tensorflow 2.4.0
* scikit-learn 0.23.1

## Quick start

From [JD JingPan](http://box.jd.com/sharedInfo/FC5D1899151CF81DEEC946AFBC5707A7), download train dataset(named *train* folder) and test dataset(named *test* folder) into *data* folder.

```bash
sh run.sh
```

## Paper Citation 

```bash
@inproceedings{10.1145/3511808.3557683,
author = {Song, Jinbo and Huang, Ruoran and Wang, Xinyang and Huang, Wei and Yu, Qian and Chen, Mingming and Yao, Yafei and Fan, Chaosheng and Peng, Changping and Lin, Zhangang and Hu, Jinghe and Shao, Jingping},
title = {Rethinking Large-Scale Pre-Ranking System: Entire-Chain Cross-Domain Models},
year = {2022},
isbn = {9781450392365},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3511808.3557683},
doi = {10.1145/3511808.3557683},
booktitle = {Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
pages = {4495–4499},
numpages = {5},
keywords = {pre-ranking, cross-domain, recommendation system},
location = {Atlanta, GA, USA},
series = {CIKM '22}
}
```
