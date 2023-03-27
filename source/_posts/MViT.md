---
title: MViT
date: 2023-03-23 21:11:34
tags: [Transformer-based,代码]
categories:
---


paper: https://arxiv.org/abs/2104.11227

code: https://github.com/facebookresearch/SlowFast

# 摘要

我们提出了用于视频和图像识别的多尺度视觉Transformer（MViT），通过将多尺度特征层次结构的开创性想法与Transformer模型相连接。多尺度Transformer具有多个通道-分辨率尺度阶段。从输入分辨率和小通道维度开始，这些阶段在减小空间分辨率的同时分层扩展通道容量。这创建了一个多尺度特征金字塔，早期层以高空间分辨率操作，以模拟简单的低层次视觉信息，而深层以空间粗糙但复杂的高维特征操作。我们评估了这个基本的架构先验来模拟视觉信号的密集性质，针对多种视频识别任务进行了评估，其中它的表现优于依赖于大规模外部预训练的并且计算和参数成本高5-10倍的同时期视觉Transformer。我们进一步去除了时间维度，并将我们的模型应用于图像分类，其中它比视觉Transformer之前的工作中表现更好。

# 方法

通用多尺度变压器架构是建立在stages得核心概念上的。每一个stage由多个具有特定时空分辨率和channel维度的Transformer块组成。主要思想是逐步扩大信道容量，同时汇集网络从输入到输出的分辨率。

## 多头池化注意力

多头池化注意力（Multi Head Pooling Attention,MHPA）是一种可以在Transformer块中灵活建模不同分辨率的自注意力算子，使得多尺度Transformer可以在不断变化的时空分辨率下运行。与原始的多头注意力算子相比，MHPA将潜在张量序列进行池化，以减少参与输入的序列长度(分辨率)，如下图。
![20230326125003](https://yic-123.oss-cn-guangzhou.aliyuncs.com//img/20230326125003.png)
输入$X \in \mathbb{R}^{L \times D}$是一个序列长度为$L$的$D$维输入张量，与MHA一样得到query、key、value：

$$
\hat{Q} = XW_{Q} \quad \hat{K} = XW_{K} \quad \hat{V} = XW_{V}
$$

**池化算子**：$\mathcal{P}(\cdot ; \mathbf{\Theta})$沿着每个维度对输入张量执行池化核计算。$\mathbf{\Theta}$内的参数为$\mathbf{\Theta} := (\mathbf{k}, \mathbf{s}, \mathbf{p})$，分别对应池化核大小、步长、padding。因此注意力 机制的计算会变为：

$$
\operatorname{PA}(\cdot) = \operatorname{Softmax}(\mathcal{P}(Q; \mathbf{\Theta}_Q)\mathcal{P}(K; \mathbf{\Theta}_K)^T/\sqrt{d})\mathcal{P}(V; \mathbf{\Theta}_V),
$$


