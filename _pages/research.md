---
layout: archive
title: "Research"
permalink: /research/
author_profile: true
---

My research is motivated by the expansion of ''big data" and the challenges introduced in these high-dimensional settings. A prominent example arises in genomics, where researchers analyze large biological datasets to understand how genetic variation influences traits and disease. In such research, DNA is extracted and sequenced, producing tens of thousands of features, each representing a gene, nucleotide, or molecular marker. 

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
Dimension reduction is often associated with the field of multivariate statistics, which is a field that can carry a reputation for being difficult due to its heavy use of matrix algebra and abstract geometric concepts. While this perception has some truth—even for me, as I often have to revisit the underlying math—dimension reduction is actually quite intuitive and something people constantly do without realizing it. In fact, this paragraph is a form of dimension reduction. That is, dimension reduction is simply the act of reducing the size of any information into a smaller and more digestible form. Ever skip through a video just to catch the main points or watch a TikTok at 2x speed? Both are examples of dimension reduction, with countless more found in everyday life. However, sometimes when watching a video at 2x speed, you may miss important details, and a slightly slower speed would be better. This illustrates the idea of **sufficient dimension reduction**, which is *the process of reducing data to its smallest possible form while still preserving all the essential information relevant to the question or goal of interest*

In mathematical terms, let $Y$ represent the response or variable of interest that we want to preserve, and let $X$ represent the predictor or data we have. Then, the goal of *SDR* is to project \(\mathbf{X}\) onto the smallest possible subspace \(\mathcal{S} \subseteq \mathbb{R}^{p}\) without any loss of information with respect to \(Y|\mathbf{X}\), which denotes the conditional distribution of $Y$ given $X$. 

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







