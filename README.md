# Estimating Individual Causal Treatment Effect by Variable Decomposition

## Introduction
Estimating individual-level causal effects is crucial for decision-making in various domains, such as personalized healthcare, social marketing, and public policy. Addressing confounding bias is a critical step in accurately estimating the causal effects of treatments on outcomes. However, many current causal inference approaches consider all observed variables as confounders without distinguishing them from colliders or indirect (two-order) colliders. This may lead to M-bias when improperly eliminating confounding bias. In this study, we propose a new framework to accurately estimate individual-level treatment effects by considering a causal structure that includes both confounding variables and indirect  colliders. Specifically, we first perform a sample reweighting to approximately eliminate confounding bias. Then, we restore the covariate’ potential latent parents and extract the modules solely related to the outcome. Finally, we take both these modules with the treatment variables to infer counterfactuals for causal inference. To validate the effectiveness of our proposed approach, we conduct extensive experiments on synthetic and commonly used semi-synthetic benchmark datasets. The experimental results demonstrate that our method outperforms current state-of-the-art methods.

## Authors

- Hongyang Jiang, Yonghe Zhao, Qiang Huang, Yangkun Cao, Huiyan Sun*, Yi Chang*


## Example command

```shell
python3 run_this.py -data="IHDP" -w="mlp"
```

## Environments

The python version is 3.6, and the main environments are as follow:
 - numpy == 1.19.2
 - pandas == 1.1.5
 - torch == 1.9.0
 - scikit-learn == 0.24.1
 - scipy == 1.5.2

## Datasets

1. IHDP
 - The dataset is download from https://github.com/Osier-Yi/SITE/tree/master/data

2. News
 - The dataset is download from https://www.fredjo.com/

3. Mdata
 - A Synthetic dataset

## Cite
If you make use of this code in your own work, please cite our paper:

```
@inproceedings{jiang2024deltanet,
  title={Estimating Individual Causal Treatment Effect by Variable Decomposition},
  author={Hongyang Jiang, Yonghe Zhao, Qiang Huang, Yangkun Cao, Huiyan Sun, and Yi Chang},
  booktitle={International Joint Conference on Neural Networks},
  year={2024}
}
```
