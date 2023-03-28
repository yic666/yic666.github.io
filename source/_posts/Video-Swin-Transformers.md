---
title: Video-Swin-Transformers
date: 2023-03-28 19:49:09
tags: [Transformer-based]
categories: [视频理解]
---

paper: https://arxiv.org/abs/2106.13230

code: https://github.com/SwinTransformer/Video-Swin-Transformer

# 摘要

视觉社区正在见证从CNN到Transformer的建模转变，其中纯Transformer架构在主要视频识别基准上获得了最高准确性。这些视频模型都建立在Transformer层上，它们在空间和时间维度上全局连接补丁。在本文中，我们反而提倡在视频变换器中引入局部性的归纳偏差，与先前计算全局自我关注甚至具有空间-时间因子分解的方法相比，这导致了更好的速度-准确性折衷。所提出的视频架构的局部性是通过适应为图像域设计的Swin Transformer实现的，同时继续利用预训练图像模型的能力。我们的方法在广泛的视频识别基准上实现了最先进的准确性，包括动作识别（Kinetics-400上84.9的top-1准确性和Kinetics-600上85.9的top-1准确性，预训练数据约少20倍，模型大小约小3倍）和时间建模（Something-Something v2上69.6的top-1准确性）。

