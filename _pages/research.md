---
layout: archive
title: "Research"
permalink: /research/
author_profile: true
---

My research is motivated by the expansion of "big data" and the challenges introduced in these high-dimensional settings. A prominent example arises in genomics, where researchers analyze large biological datasets to understand how genetic variation influences traits and disease. In such research, DNA is extracted and sequenced, producing tens of thousands of features, each representing a gene, nucleotide, or molecular marker. 

<div style="text-align:center;">

  <img src="/files/dna.png" alt="DNA sequence visualization" style="width:40%; border-radius:8px;">

  <div style="font-size:0.85em; color:gray;">
    Source:
    <a href="https://medium.com/@xiaofan.luan/dna-sequence-classification-based-on-milvus-f87e87bc5ba9"
       target="_blank" style="color:gray; text-decoration:none;">
       Medium — Xiaofan Luan
    </a>
  </div>

</div>

Reducing this immense dimensionality of features to capture the essential biological signal is critical for revealing meaningful genetic structure, improving disease prediction, and developing targeted therapies. My research addresses these issues by developing sufficient dimension reduction methodologies that extract the most informative structures from high-dimensional data and project them onto lower-dimensional subspaces for improved modeling, inference, and visualization.

## Sufficient Dimension Reduction
--- 
### Basics
Dimension reduction is often associated with the field of multivariate statistics, which is a field that can carry a reputation for being difficult due to its heavy use of matrix algebra and abstract geometric concepts. While this perception has some truth—even for me, as I often have to revisit the underlying math—dimension reduction is actually quite intuitive and something people constantly do without realizing it. In fact, this paragraph is a form of dimension reduction. That is, dimension reduction is simply the act of reducing the size of any information into a smaller and more digestible form. Ever skip through a video just to catch the main points or watch a TikTok at 2x speed? Both are examples of dimension reduction, with countless more found in everyday life. However, sometimes when watching a video at 2x speed, you may miss important details, and a slightly slower speed would be better. This illustrates the idea of **sufficient dimension reduction** (*SDR*), which is **the process of reducing data to its smallest possible form while still preserving all the essential information relevant to the question or goal of interest**.

### Formal Definition \& Central Subspace
In mathematical terms, let $Y$ represent the response or variable of interest that we want to preserve, and let $\mathbf{X} \in \mathbb{R}^{p}$ represent the predictor or data we have with $p$ features. Then, the goal of *SDR* is to project $\mathbf{X}$ onto the smallest possible subspace $\mathcal{S} \subseteq \mathbb{R}^{p}$ without any loss of information with respect to $Y \mid \mathbf{X}$, which denotes the conditional distribution of $Y$ given $\mathbf{X}$ .  So how do we get this smallest possible subspace? [Cook (1998)](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316931) introduced  the central subspace  $\mathcal{S}\_{Y \mid \mathbf{X}}$, which is the intersection of all possible dimension reduction subspaces. Thus, $\mathcal{S}\_{Y \mid \mathbf{X}}$  is the unique and minimal dimension reduction subspace that preserves all the information about $Y$ given $\mathbf{X}$.  Therefore, most SDR methods developed are trying to estimate $\mathcal{S}\_{Y \mid \mathbf{X}}$. Let $d < p$ be the dimension of $\mathcal{S}\_{Y \mid \mathbf{X}}$ and $\boldsymbol{\beta} = \left(\boldsymbol{\beta}\_{1}, \ldots, \boldsymbol{\beta}\_{d}\right)$ be a basis matrix of $\mathcal{S}\_{Y\mid \mathbf{X}}$ determined by the particular *SDR* technique.  Then, *SDR* methods can reduce the dimensionality of the features to $\boldsymbol{\beta}^{\top}\mathbf{X} \in \mathbb{R}^{d}$ for subsequent supervised learning without loss of information. 

### Eigenvectors for Dimension Reduction
So, how do we determine the projection matrix $\boldsymbol{\beta} \in \mathbb{R}^{p \times d}$ used in *SDR* to reduce the data from $p$ dimensions to $d < p$ dimensions? Most *SDR* methods determine $\boldsymbol{\beta}$ by using eigenvectors (if you already understand eigenvectors, feel free to skip below). Suppose we have data given by the matrix $\mathbf{A} \in \mathbb{R}^{p \times p}$, then eigenvectors are any vector $\mathbf{v} \in \mathbb{R}^{p}$ that satisfy $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$, where $\lambda$ is a scaler value. Most encounter this concept in a linear algebra course and learn how to numerically solve for $\mathbf{v}$. However, I find it is not always clear to most what an eigenvector represents beyond its formulation, or at least for me it initially wasn't. So, I'll try my best to give a visual representation that makes the idea more intuitive. As a reminder, in dimension reduction we aim to transform our data into a smaller form. Thus, if we think about our data matrix $\mathbf{A}$ and an arbitrary vector $\boldsymbol{\omega}$, multiplying $\mathbf{A} \boldsymbol{\omega}$ applies a linear transformation that can rotate, stretch, or compress the data in space. We show this transformation below for a typical vector $\boldsymbol{\omega}$ and an eigenvector $\mathbf{v}$. 

<div style="display:flex; justify-content:center; align-items:flex-start; gap:10px;">

  <div style="text-align:center; flex:1;">
    <div style="font-weight:600; margin-bottom:6px;">A typical vector ($\mathbf{\omega}$)</div>
    <img src="/files/omega.gif" alt="Omega animation" style="width:80%; border-radius:8px;">
  </div>

  <div style="text-align:center; flex:1;">
    <div style="font-weight:600; margin-bottom:6px;">An eigenvector ($\mathbf{v}$)</div>
    <img src="/files/v.gif" alt="V animation" style="width:80%; border-radius:8px;">
  </div>

</div>

In the figures above, the grid represents the transformation, and the dotted line (red) shows all possible points that lie along the same direction as the vector, i.e., the span of the vector. We see that the typical vector $\boldsymbol{\omega}$ is knocked off its span after the transformation, which means its direction changes and it no longer preserves the same information in the data. In contrast, the eigenvector $\mathbf{v}$ only stretches slightly. The amount by which the eigenvector stretches is its eigenvalue $\lambda$. Hence, the eigenvector $\mathbf{v}$ stays on its span during the transformation and thereby represents the principal directions of the data matrix $\mathbf{A}$. If we find all eigenvectors of $\mathbf{A}$, they form the coordinate system aligned with how $\mathbf{A}$ transforms space. Therefore, by using $d$ of the eigenvectors to form the basis for the projection matrix, we can reduce the data onto these principal directions and effectively capture the dominant structure of $\mathbf{A}$ in this lower $d$-dimensional space.

### Generalized Eigenvalue Problem
Most SDR methods do not simply use the eigenvectors of the data matrix $\mathbf{X}$ to construct the projection matrix. Why? Because doing so still involves the entire dataset, which can be extremely large, computationally expensive, and contains a lot of noise. Instead, it is often more practical to work with summary measures of the data, such as means and covariances. Moreover, in *SDR* we are specifically concerned with preserving all information in the response $Y$ with respect to the predictor $\mathbf{X}$. Thus, most *SDR* methods construct a method-specific kernel matrix $\mathbf{M} \in \mathbb{R}^{p \times p}$ that captures the relationship between $Y$ and $\mathbf{X}$. For example, [Li (1991)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1991.10475035) introduced this concept through the *sliced inverse regression* (*SIR*) method by proposing the kernel matrix  $\mathbf{M}_{SIR} = \text{Cov}\left(\mathbb{E}[\mathbf{X} - \mathbb{E}\{\mathbf{X}|Y\}]\right)\) & \(\boldsymbol{\Sigma}_{\mathbf{X}}$. 

## Dimension Reduction Subspace Criteria
--- 




## Publications & Manuscripts
---

**Boonstra, D. T.**, Kim, R., and Young, D. M. (2025).  
“Subspace Ordering for Maximum Response Preservation in Sufficient Dimension Reduction.”  
*Submitted*

**Boonstra, D. T.**, Kim, R., and Young, D. M. (2025).  
“Precision Matrix Regularization in Sufficient Dimension Reduction for Improved Quadratic Discriminant Classification.”  
*Under Review*  doi: [10.48550/arXiv.2506.19192](https://doi.org/10.48550/arXiv.2506.19192)

### Manuscripts in Progress

**Boonstra, D. T.**, Kim, R., and Young, D. M.  
“Heteroscedastic Invariant Sufficient Dimension Reduction.”

**Boonstra, D. T.**, Kim, R., and Young, D. M.  
“Sufficient Dimension Reduction Methods for High-Dimensional Data Are Overly Complicated.”

**Boonstra, D. T.**, Kim, R., and Young, D. M.  
“Ordering Dimension Reduction Subspaces via a Quadratic Discriminant Optimal Error Rate.”







