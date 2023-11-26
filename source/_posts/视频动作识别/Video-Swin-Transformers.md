---
title: Video-Swin-Transformers
tags:
  - Transformer-based
categories:
  - 视频动作识别
date: 2023-03-28 19:49:09
---

[paper](https://arxiv.org/abs/2106.13230)  [code](https://github.com/SwinTransformer/Video-Swin-Transformer)

# 摘要

视觉社区正在见证从CNN到Transformer的建模转变，其中纯Transformer架构在主要视频识别基准上获得了最高准确性。这些视频模型都建立在Transformer层上，它们在空间和时间维度上全局连接补丁。在本文中，我们反而提倡在视频变换器中引入局部性的归纳偏差，与先前计算全局自我关注甚至具有空间-时间因子分解的方法相比，这导致了更好的速度-准确性折衷。所提出的视频架构的局部性是通过适应为图像域设计的Swin Transformer实现的，同时继续利用预训练图像模型的能力。我们的方法在广泛的视频识别基准上实现了最先进的准确性，包括动作识别（Kinetics-400上84.9的top-1准确性和Kinetics-600上85.9的top-1准确性，预训练数据约少20倍，模型大小约小3倍）和时间建模（Something-Something v2上69.6的top-1准确性）。

<!--more-->

# 方法

## 总体架构

Video Swin Transformer的总体架构如图所示，它展示了其tiny版本（Swin-T）。输入视频被定义为大小为$T×H×W×3$，由$T$帧组成，每帧包含$H×W×3$个像素。在Video Swin Transformer中，将每个大小为$2\times 4 \times 4 \times 3$的3D块视为一个token。因此，3D patch partitioning layer获得$\frac{T}{2}×\frac{H}{4}×\frac{W}{4}$个3D标记，每个块/token由96维特征组成。然后应用线性embedding层将每个标记的特征投影到一个任意维度$C$。

根据先前的工作，时间维度不进行下采样，这样就能直接参考Swin Transformer的架构，包含4个阶段，在每个阶段的Patch Merging层执行$2\times$空间下采样（将每组2 $\times$ 2空间相邻补丁的特征拼接起来，并应用一个线性层将拼接后的特征投影到其维数的一半）。

该架构的主要组成部分是视频Swin Transformer块，该块是通过将标准Transformer层中的多头自注意(MSA)模块替换为基于3D移动窗口的多头自注意模块，并保持其他组件不变的方式构建的。

![Video Swin Transfromer](https://yic-123.oss-cn-guangzhou.aliyuncs.com//img/20230329134824.png)

## 基于3D移位窗口的MSA模块

![3D移位窗口](https://yic-123.oss-cn-guangzhou.aliyuncs.com//img/20230329145740.png)

**非重叠3D窗口的多头注意力机制**：直接扩展2D的方法去处理视频输入。给定一个由$T’$$\times$$H’$$\times$$W’$ 3D token组成的视频和一个$P$$\times$$M$$\times$$M$的3D窗口大小，窗口被安排以非重叠的方式均匀划分视频输入。也就是说，输入token被划分为$\lceil\frac{T’}{P}\rceil$$\times$$\lceil\frac{H’}{M}\rceil$$\times$$\lceil\frac{W’}{M}\rceil$个不重叠的3D窗口。例如，如图所示，对于输入大小为$8\times8\times8$个token和窗口大小为$4\times 4 \times 4$的情况，第$l$层的窗口数量将为$2\times 2 \times 2$=8,并且在每个3D窗口内执行多头自注意力。

**3D移位窗口**：同样，类似于Swin Transformer，不同窗口之间缺乏连接，因此将Swin Transformer的2D移位窗口机制扩展到3D，使得可以实现跨窗口的信息连接。鉴于输入3D令牌的数量为$T’$$\times$$H’$$\times$$W’$，每个3D窗口的大小为$P$$\times$$M$$\times$$M$，对于两个连续的层，第一层中的自注意力模块使用常规窗口划分策略，第二层则是把窗口划分配置沿时间轴、高度轴和宽度轴分别移动($\frac{P}{2}$,$\frac{M}{2}$,$\frac{M}{2}$)个token，也就是相比于Swin Transformer多了时间轴的移动。

通过移动窗口划分方法，两个连续的视频Swin变换器块被计算为

$$
    { {\hat{\bf{z} } }^{l} } = \text{3DW-MSA}\left( {\text{LN}\left( { { {\bf{z} }^{l - 1} } } \right)} \right) + {\bf{z} }^{l - 1},\\\\
    { {\bf{z} }^l} = \text{FFN}\left( {\text{LN}\left( { { {\hat{\bf{z} } }^{l} } } \right) } \right) + { {\hat{\bf{z} } }^{l} },\\\\
    { {\hat{\bf{z} } }^{l+1} } = \text{3DSW-MSA}\left( {\text{LN}\left( { { {\bf{z} }^{l} } } \right)} \right) + {\bf{z} }^{l}, \\\\
    { {\bf{z} }^{l+1} } = \text{FFN}\left( {\text{LN}\left( { { {\hat{\bf{z} } }^{l+1} } } \right)} \right) + { {\hat{\bf{z} } }^{l+1} }, 
$$

其中，${\hat{\bf{z}}}^l$和${\bf{z}}^l$分别表示第$l$块的3D(S)W-MSA模块和FFN模块的输出特征；$\text{3DW-MSA}$和$\text{3DSW-MSA}$分别表示使用常规和移动窗口划分配置的基于3D窗口的多头自注意力。

**3D相对位置偏置**：与大多数工作类似，作者也在每个注意力头加了3D相对位置偏置$B \in \mathbb{R}^{P^2 \times M^2 \times M^2}$如下

$$
    \text{Attention}(Q, K, V) = \text{SoftMax}(QK^T/\sqrt{d}+B)V,
$$

