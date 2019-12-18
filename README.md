# Calibrated Domain-Invariant Learning for Highly Generalizable Large Scale Re-Identification

<a href="https://arxiv.org/abs/1911.11314">Calibrated Domain-Invariant Learning for Highly Generalizable Large Scale Re-Identification</a>

Ye Yuan, Wuyang Chen, Tianlong Chen, Yang Yang, Zhou Ren, Zhangyang Wang, Gang Hua

In WACV 2020.

## Overview

Many real-world applications, such as city-scale traffic monitoring and control, requires large-scale re-identification. However, previous ReID methods often failed to address two limitations in existing ReID benchmarks, i.e., low spatiotemporal coverage and sample imbalance. Notwithstanding their demonstrated success in every single benchmark, they have difficulties in generalizing to unseen environments. As a result, these methods are less applicable in a large-scale setting due to poor generalization. 

In seek for a highly generalizable large-scale ReID method, we present an adversarial domain invariant feature learning framework (ADIN) that **explicitly learns to separate identity-related features from challenging variations**, where for the first time "free" annotations in ReID data such as video timestamp and camera index are utilized. 

Furthermore, we find that the imbalance of nuisance classes jeopardizes the adversarial training, and for mitigation we propose a calibrated adversarial loss that is attentive to nuisance distribution. Experiments on existing large-scale person vehicle ReID datasets demonstrate that ADIN learns more robust and generalizable representations, as evidenced by its outstanding direct transfer performance across datasets, which is a criterion that can better measure the generalizability of large-scale ReID methods.

## Methods

<p align="center">
<img src="https://raw.githubusercontent.com/TAMU-VITA/ADIN/master/figures/adv_model.png" alt="ADIN" width="600"/></br>
<b>Adversarial Model</b>
</p>

<p align="center">
<img src="https://github.com/TAMU-VITA/ADIN/blob/master/figures/dual-branch.png" alt="ADIN" width="600"/></br>
<b>Dual Branch Structure</b>
</p>

## Training

Please sequentially finish the following steps:
1. `python script/train.py --dataset MSMT17 --loss crossEntropy` (save checkpoint)
1. `python script/train.py --dataset MSMT17 --loss classCamIdAndTimeStamp --resume-checkpoint timestamp`
1. 

## Evaluation

Run script
1. `python script/featureExtract.py`
1. `python script/evaluate.py`


## Citation

If you use this code for your research, please cite our paper.
```
@inproceedings{yuan2020ADIN,
  title={Collaborative Global-Local Networks for Memory-Efﬁcient Segmentation of Ultra-High Resolution Images},
  author={Ye Yuan, Wuyang Chen, Tianlong Chen, Yang Yang, Zhou Ren, Zhangyang Wang, Gang Hua},
  booktitle={WACV},
  year={2020}
}
```
