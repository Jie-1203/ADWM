# A General Adaptive Dual-level Weighting Mechanism for Remote Sensing Pansharpening [CVPR 2025]

<!-- <div style="text-align: center;">
  <a href="https://www.arxiv.org/">
    <img src="https://img.shields.io/badge/arXiv-red.svg?style=flat" alt="ArXiv">
  </a>
    <a href="https://arxiv.org/abs/2404.11537">Arxiv</a>
    <!-- <img class="nips-logo" src="https://neurips.cc/static/core/img/NeurIPS-logo.svg" alt="NeurIPS logo" height="25" width="58">
    <a href="https://openreview.net/pdf?id=QMVydwvrx7">NeurIPS 2024 -->
</div> 
<p style="text-align: center; font-family: 'Times New Roman';">
  </a>
    Abstract:
    Currently, deep learning-based methods for remote sensing pansharpening have advanced rapidly. However, many existing methods struggle to fully leverage feature heterogeneity and redundancy, thereby limiting their effectiveness. 
We use the covariance matrix to model the feature heterogeneity and redundancy and propose Correlation-Aware Covariance Weighting (CACW) to adjust them. CACW captures these correlations through the covariance matrix, which is then processed by a nonlinear function to generate weights for adjustment. Building upon CACW, we introduce a general adaptive dual-level weighting mechanism (ADWM) to address these challenges from two key perspectives, enhancing a wide range of existing deep-learning methods. First, Intra-Feature Weighting (IFW) evaluates correlations among channels within each feature to reduce redundancy and enhance unique information. Second, Cross-Feature Weighting (CFW) adjusts contributions across layers based on inter-layer correlations, refining the final output. Extensive experiments demonstrate the superior performance of ADWM compared to recent state-of-the-art (SOTA) methods. Furthermore, we validate the effectiveness of our approach through generality experiments, redundancy visualization, comparison experiments, key variables and complexity analysis, and ablation studies.
</a>
</p>

News:
- 2025/2/7：**Code RELEASED!**:fire: 

- 2025/3/12: **Code will be released soon!**:fire: 

## Quick Overview

The code in this repo supports Pansharpening.

<center>
<img src="figs/WFANet.png" width="80%">
<img src="figs/MFFA.png" width="80%">
</center>

# Instructions

## Dataset

In this office repo, you can find the Pansharpening dataset of [WV3, GF2, and QB](https://github.com/liangjiandeng/PanCollection).

Other instructions will come soon!

<!-- ## Citation

If you find our paper useful, please consider citing the following: -->

<!-- ```tex
@article{zhong2024ssdiff,
  title={SSDiff: Spatial-spectral Integrated Diffusion Model for Remote Sensing Pansharpening},
  author={Zhong, Yu and Wu, Xiao and Deng, Liang-Jian and Cao, Zihan},
  journal={arXiv preprint arXiv:2404.11537},
  year={2024}
}
``` -->
