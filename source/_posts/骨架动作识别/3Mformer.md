---
title: 3Mformer
tags:
  - 论文笔记
  - 动作识别
  - CVPR2023
  - Transformer-based
  - 骨架
categories:
  - 骨架动作识别
date: 2023-07-18 12:53:38
---

paper: https://arxiv.org/abs/2303.14474

# 摘要

许多骨骼动作识别模型使用图卷积网络（GCNs）通过连接身体部位的三维关节来表示人体。GCNs聚合一或少数跳的图邻域，并忽略未连接的身体关节之间的依赖关系。我们提出使用超图来建模图节点之间的超边（例如，三阶和四阶超边捕捉三个和四个节点），从而帮助捕捉身体关节组的高阶运动模式。我们将动作序列分割为时间块，Higher-order Transformer（HoT）根据（i）身体关节，（ii）身体关节之间的成对连接，以及（iii）骨骼身体关节的高阶超边，生成每个时间块的嵌入。我们通过一种新颖的多阶多模Transformer（3Mformer）结合这些超边的HoT嵌入，该Transformer具有两个模块，可以交换顺序，实现基于“通道-时间块”、“顺序-通道-身体关节”、“通道-超边（任意阶）”和“仅通道”对上的耦合模式注意力。第一个模块称为多阶汇聚（MP），还可以学习沿着超边模式的加权汇聚，而第二个模块称为时间块汇聚（TP），则沿着时间块1模式进行汇聚。我们的端到端可训练网络相对于基于GCN、Transformer和超图的对应方法获得了最先进的结果。

<!--more-->

# 引言

论文背景: 骨骼动作识别在视频监控、人机交互、体育分析和虚拟现实等领域具有广泛应用。与基于视频的方法不同，骨骼序列通过表示3D身体关节的时空演变，对传感器噪声具有鲁棒性，并且在计算和存储效率上更高效。

过去方案: 过去的图形模型主要通过图卷积网络（GCN）或图神经网络（GNN）来处理骨骼数据。然而，这些方法忽略了非连接的关节之间的依赖关系，并且对于捕捉更高阶的运动模式有限。

论文的Motivation: 鉴于现有方法的局限性，本研究旨在提出一种新的模型来更好地表示骨骼数据，并捕捉关节之间的高阶动态。通过构建超图来表示骨骼数据，并使用多阶多模态变压器进行耦合模式注意力，以实现对不同模态的关注。

论文的Contribution：

1. 将骨骼数据建模为阶数为1到r的超图（集合、图和/或超图），其中人体关节作为节点。这样形成的超边的Higher-order Transformer嵌入表示了各种3D身体关节的组合，并捕捉了对于动作识别非常重要的各种高阶动态。
2. 由于HoT嵌入表示了各个超边的阶数和时间块，引入了一种新颖的Multi-order Multi-mode Transformer (3Mformer)。它包含两个模块，即Multi-order Pooling和Temporal block Pooling，其目标是形成诸如'通道-时间块'、'顺序-通道-身体关节'、'通道-超边（任意阶）'和'仅通道'等耦合模式tokens，并进行加权超边聚合和时间块聚合。

# 背景

## 符号表示
$\mathcal{I}_{K}$代表索引集合${1,2,\cdots,K}$。大写粗体符号表示矩阵（二阶张量）或高阶张量（超过两个模式）。小写粗体符号表示向量，普通字体表示标量。
$\mathcal{I}_{K}$代表索引集合${1,2,\cdots,K}$。普通字体表示标量；向量用小写粗体字母表示，例如$\textbf{x}$；矩阵用大写粗体字母表示，例如$\textbf{M}$；张量用花体字母表示，例如$\vec{\mathcal{M}}$。
$r$阶张量表示为$\vec{\mathcal{M}} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_r}$，
$\vec{\mathcal{M}}$的第$m$模式矩阵化表示为$\vec{\mathcal{M}}_{(m)}\in \mathbb{R}^{I_m \times (I_1 \cdots I_{m-1} I_{m+1} \cdots I_{r})}$。

## Transformer
Transformer编码器层$f : \mathbb{R}^{J \times d} \rightarrow \mathbb{R}^{J \times d}$包含两个子层：(i) 自注意力$a : \mathbb{R}^{J \times d} \rightarrow \mathbb{R}^{J \times d}$和(ii) 逐元素的前馈网络$\text{MLP} :\mathbb{R}^{J \times d} \rightarrow \mathbb{R}^{J \times d}$。对于具有$J$个节点的集合，其中${\bf X}  \in \mathbb{R}^{J \times d}$，${\bf x}_i$是节点$i$的特征向量，一个Transformer层(为简洁起见，省略了$a(\cdot)$和MLP$(\cdot)$后的归一化)计算如下：
$$
\begin{align}
    & a({\bf x}_i) = {\bf x}_i + \sum_{h=1}^H\sum_{j=1}^J\alpha_{ij}^h{\bf x}_j{\bf W}_h^V{\bf W}_h^O, \\
    & f({\bf x}_i) = a({\bf x}_i) + \text{MLP}(a({\bf X}))_i, 
\end{align}
$$
其中$H$和$d_H$分别表示注意力头的数量和头的大小，${\boldsymbol{\alpha}}^h = \sigma\big({\bf X}{\bf W}_h^Q({\bf X}{\bf W}_h^{K})^\top\big)$是注意力系数，${\bf W}_h^O \in \mathbb{R}^{d_H \times d}$，${\bf W}_h^V$，${\bf W}_h^K$，${\bf W}_h^Q  \in \mathbb{R}^{d \times d_H}$。

## Higher-order transformer

设HoT层为$f_{m\rightarrow n} :\mathbb{R}^{J^m  \times d} \rightarrow \mathbb{R}^{J^n \times d}$，其中包含两个子层：(i) 高阶自注意力$a_{m\rightarrow n} :\mathbb{R}^{J^m  \times d} \rightarrow \mathbb{R}^{J^n \times d}$和(ii) 前馈网络$\text{MLP}_{n\rightarrow n} :\mathbb{R}^{J^n  \times d} \rightarrow \mathbb{R}^{J^n \times d}$。此外，引入索引向量${\bf i}\in\mathcal{I}_{J}^m\equiv\mathcal{I}_{J} \times \mathcal{I}_{J} \times \cdots \times \mathcal{I}_{J}$（$m$个模式）和${\bf j}\in\mathcal{I}_{J}^n\equiv\mathcal{I}_{J} \times \mathcal{I}_{J} \times \cdots \times \mathcal{I}_{J}$（$n$个模式）。对于输入张量${\bf X} \in \mathbb{R}^{J^m \times d}$，其中超边的阶数为$m$，HoT层的计算如下：

$$
\begin{align}
    & a_{m \rightarrow n}(\mathbf{X})_{j}=\sum_{h=1}^{H} \sum_{\mu} \sum_{i} \alpha_{i, j}^{h, \mu} \mathbf{X}_{i} \mathbf{W}_{h, \mu}^{V} \mathbf{W}_{h, \mu}^{o} \\
    & \operatorname{MLP}_{n \rightarrow n}\left(a_{m \rightarrow n}(\mathbf{X})\right)=\mathrm{L}_{n \rightarrow n}^{2}\left(\operatorname{ReLU}\left(\mathrm{L}_{n \rightarrow n}^{1}\left(a_{m \rightarrow n}(\mathbf{X})\right)\right)\right), \\
    & f_{m \rightarrow n}(\mathbf{X})=a_{m \rightarrow n}(\mathbf{X})+\operatorname{MLP}_{n \rightarrow n}\left(a_{m \rightarrow n}(\mathbf{X})\right), 
\end{align}
$$
其中${\boldsymbol \alpha}^{h, \mu} \in \mathbb{R}^{J^{m+n}}$是具有多个头部的所谓注意力系数张量，${\boldsymbol \alpha}^{h, \mu}_{\mathbf{i},\mathbf{j}} \in \mathbb{R}^{J}$是一个向量，${\bf W}_{h, \mu}^V \in \mathbb{R}^{d \times d_H}$和${\bf W}_{h, \mu}^O \in \mathbb{R}^{d_H \times d}$是可学习的参数。此外，$\mu$在相同节点分区中的阶-$(m+n)$的等价类上进行索引，$\text{L}_{n\rightarrow n}^1 :\mathbb{R}^{J^n \times d}\rightarrow \mathbb{R}^{J^n \times d_F}$和$\text{L}_{n\rightarrow n}^2 :\mathbb{R}^{J^n \times d_F}\rightarrow \mathbb{R}^{J^n \times d}$是等变线性层，$d_F$是隐藏维度。

为了从阶数为$m$的输入张量${\bf X} \in \mathbb{R}^{J^m \times d}$中计算每个注意力张量${\boldsymbol \alpha}^{h,\mu} \in \mathbb{R}^{J^{m+n}}$，根据高阶query和key，我们有：
$$
\begin{equation}
  {\boldsymbol{\alpha}_{\boldsymbol i, \boldsymbol j}^{h,\mu}}  = 
    \begin{cases}
      \frac{\sigma({\bf Q}_{\boldsymbol j}^{h,\mu}, {\bf K}_{\boldsymbol i}^{h,\mu})}{Z_{\boldsymbol j}}\;\quad({\boldsymbol i}, {\boldsymbol j})  \in  \mu\\
      \quad\quad 0 \quad\quad\;\text{otherwise},
    \end{cases}
    
\end{equation}
$$
其中${\bf Q}^\mu = \text{L}_{m\rightarrow n}^\mu({\bf X})$，${\bf K}^\mu = \text{L}_{m\rightarrow m}^\mu({\bf X})$，归一化常数$Z_{\boldsymbol j} = \sum_{\boldsymbol i:({\boldsymbol i}, {\boldsymbol j})\in \mu}\sigma({\bf Q}_{\boldsymbol j}^\mu, {\bf K}_{\boldsymbol i}^\mu)$。最后，可以将Eq(6)中的核注意力近似为具有RKHS特征映射$\psi\in\mathbb{R}_{+}^{d_K}$以提高效率，其中$d_K\ll d_H$。具体而言，有$\sigma({\bf Q}_{\boldsymbol j}^{h,\mu}, {\bf K}_{\boldsymbol i}^{h,\mu})\approx{\boldsymbol \psi}({\bf Q}_{\boldsymbol j}^{h,\mu})^\top{\boldsymbol \psi}({\bf K}_{\boldsymbol i}^{h,\mu})$。选择了performer核，因为它在理论和实证上都有保证。

由于query和key张量是使用等变线性层从输入张量${\bf X}$计算得到的，因此Transformer编码器层$f_{m\rightarrow n}$满足排列等变性。

# 方法

## 模型概览

![](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/image.png)

如图所示，框架包含一个简单的三层MLP单元（全连接层，ReLU激活函数，全连接层，ReLU激活函数，Dropout层，全连接层），三个针对每种输入类型（即，身体关节特征集、身体关节的图和超图）的HoT块，接着是具有两个模块（i）多阶池化（MP）和（ii）时间块池化（TP）的多阶多模态Transformer（3Mformer）。

3Mformer的目标是形成耦合模式tokens（稍后会解释其含义），例如“通道-时间块”、“序号-通道-身体关节”、“通道-超边（任意序号）”和“仅通道”，并进行加权超边聚合和时间块聚合。它们的输出进一步连接并传递到一个全连接层进行分类。

**MLP 单元**:MLP单元接受$T$个相邻的帧，每个帧具有$J$个2D/3D骨骼身体关节，形成一个时间块。总共，取决于步长$S$，我们得到一些$\tau$个时间块（一个块捕获短期时间演变）。相比之下，长期时间演变由HoT和3Mformer建模。每个时间块由MLP编码成一个$d\times J$维的特征图。

**HoT 分支**：我们将 HoT（Hypergraph on Transformer） 的 $r$ 个分支堆叠在一起，每个分支接收维度为 ${\bf X}_t\in\mathbb{R}^{d \times J}$ 的嵌入，其中 $t\in\mathcal{I}_{\tau}$ 表示时间块。每个 HoT 分支输出大小为 $m\in\mathcal{I}_{r}$ 的超边特征表示，记为 ${\bf\Phi}'_m\in\mathbb{R}^{J^m \times d'}$，其中 $m\in\mathcal{I}_{r}$ 表示阶数。

对于一阶、二阶和更高阶的流输出 ${\bf\Phi}'_1,\cdots,{\bf\Phi}'_r$，我们进行以下步骤：(i) 交换特征通道和超边模式，(ii) 提取张量的上三角部分，然后在块-时间模式上进行连接，这样我们得到 ${\bf\Phi}_m\in\mathbb{R}^{d'\times N_{E_m}\times\tau}$，其中 $N_{E_m} = \binom{J}{m}$。随后，我们沿着超边模式连接 ${\bf\Phi}_1,\cdots,{\bf\Phi}_r$，得到一个多阶特征张量 $\vec{\mathcal{M}} \in \mathbb{R}^{d' \times N \times \tau}$，其中所有阶数的超边总数为 $N=\sum_{m=1}^r\binom{J}{m}$。

**3Mformer**：我们使用具有耦合模式自注意力（CmSA）的多阶多模式Transformer（3Mformer）来融合多阶特征张量 $\vec{\mathcal{M}}$ 中的信息流，并最终将3Mformer的输出传递给分类器进行分类。

## Coupled-mode Self-Attention

### 耦合模式tokens(Coupled-mode tokens)
我们受到标准Vision Transformer（ViT）中单类别tokens的注意区域的启发，这些区域可以用来形成一个与类别无关的本地化映射(参考 https://zhuanlan.zhihu.com/p/481304916)。研究了Transformer模型是否也能够有效地捕捉耦合模式注意力，用于更具有区分性的分类任务，例如通过学习Transformer内的耦合模式tokens来进行基于张量骨架的动作识别。为此，提出了一个多阶多模式Transformer（3Mformer），它使用耦合模式tokens来共同学习通道模式、块-时间模式、身体关节模式和阶数模式之间的各种高阶运动动态。3Mformer能够成功地从CmSA机制中生成对应于不同tokens的耦合模式关系。接下来，介绍CmSA机制。

给定阶数为 $r$ 的张量 $\vec{\mathcal{M}} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_r}$，为了形成耦合模式tokens，我们对 $\vec{\mathcal{M}}$ 进行模式-$m$的矩阵化，得到 $\textbf{M} \equiv \vec{\mathcal{M}}_{(m)}^\top \in \mathbb{R}^{(I_1 \cdots I_{m-1}  I_{m+1}  \cdots I_{r}) \times I_m}$，然后从 $\textbf{M}$ 形成耦合tokens。

举例来说，对于一个给定的三阶张量，它具有特征通道模式、超边模式和时间块模式，我们可以形成以下tokens对：

1. `channel-temporal block'：特征通道-时间块对
2. `channel-hyper-edge (any order)'：特征通道-超边 (任意阶数)对
3. `channel-only'：仅特征通道对
   
另外，如果给定的张量被用作输入并输出一个产生新模式（例如，身体关节模式）的新张量，我们可以形成以下tokens：

`order-channel-body joint'：阶数-特征通道-身体关节对

在接下来的部分，为了简化起见，使用“reshape”来进行张量的矩阵化，以形成不同类型的联合模式令牌。

联合模式自注意力（JmSA）定义如下：
$$
\begin{equation}
      a(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\text{SoftMax}\left(\frac{\mathbf{Q K}^{\top}}{\sqrt{d_{K}}}\right) \mathbf{V}
\end{equation}
$$

其中，$\sqrt{d_{K}}$ 是缩放因子，${\bf Q} = {\bf W}^q{\bf M}$，${\bf K} = {\bf W}^k{\bf M}$ 和 ${\bf V} = {\bf W}^v{\bf M}$ 分别是查询（query）、键（key）和值（value）向量，而 $\textbf{M} \equiv \vec{\mathcal{M}}{(m)}^\top$。此外，${\bf Q}$、${\bf K}$、${\bf V} \in \mathbb{R}^{(I_1 \cdots I_{m-1} I_{m+1} \cdots I_{r}) \times I_m}$，${\bf W}^q$、${\bf W}^k$、${\bf W}^v \in \mathbb{R}^{(I_1 \cdots I_{m-1} I_{m+1} \cdots I_{r}) \times (I_1 \cdots I_{m-1} I_{m+1} \cdots I_{r})}$ 是可学习的权重。我们注意到不同的联合模式标记具有不同的“注意力焦点”机制，并且我们在我们的3Mformer中应用它们来融合多阶特征表示。

## Multi-order Multi-mode Transformer

### Multi-order Pooling (MP) Module

**CmSA in MP**：我们将多阶特征表示$\vec{\mathcal{M}} \in \mathbb{R}^{d' \times N \times \tau}$重塑为${\bf M} \in \mathbb{R}^{d'\tau \times N}$（或者将后面TP中解释的输出重塑为${\bf M}' \in \mathbb{R}^{d' \times N}$），从而使模型能够关注不同类型的特征表示。让我们简单地记$d'' = d'\tau$（或者$d'' = d'$，具体取决于输入的来源）。我们形成了一个耦合模式的self-attention（如果$d''=d'\tau$，我们有"channel-temporal block"的token；如果$d''=d'$，我们有"channel-only"的token）。
$$
\begin{equation}
  a_{\mathrm{MP}}\left(\mathbf{Q}_{\mathrm{MP}}, \mathbf{K}_{\mathrm{MP}}, \mathbf{V}_{\mathrm{MP}}\right)=\operatorname{SoftMax}\left(\frac{\mathbf{Q}_{\mathrm{MP}} \mathbf{K}_{\mathrm{MP}}^{\top}}{\sqrt{d_{K_{\mathrm{MP}}}}}\right) \mathbf{V}_{\mathrm{MP}} \text {, }
\end{equation}
$$

其中，$\sqrt{d_{K_\text{MP}}}$ 是缩放因子，${\bf Q}_\text{MP}\!=\!{\bf W}_\text{MP}^q{\bf M}$，${\bf K}_\text{MP}\!=\!{\bf W}_\text{MP}^k{\bf M}$ 和 ${\bf V}_\text{MP}\!=\!{\bf W}_\text{MP}^v{\bf M}$（我们可以使用 ${\bf M}$ 或者 ${\bf M}'$）分别是查询、键和值。此外，${\bf Q}_\text{MP}$，${\bf K}_\text{MP}$，${\bf V}_\text{MP}\!\in\! \mathbb{R}^{d''\times N}$ 和 ${\bf W}_\text{MP}^q$，${\bf W}_\text{MP}^k$，${\bf W}_\text{MP}^v\!\in\! \mathbb{R}^{d''\times d''}$ 是可学习的权重。方程式(8)是一种自注意层，它基于所谓的耦合模式令牌的 ${\bf Q}_\text{MP}$ 和 ${\bf K}_\text{MP}$ 令牌嵌入之间的相关性对 ${\bf V}_\text{MP}$ 进行重新加权。

**Weighted pooling**:(8)中的注意力层产生特征表示 ${\bf O}_\text{MP}\!\in\! \mathbb{R}^{d''\times N}$，以增强例如特征通道与身体关节之间的关系。随后，我们通过对多个阶数 $m\in\mathcal{I}_{r}$ 的超边进行加权池化来处理多个阶数的超边的影响：
$$
\mathbf{O}_{\mathrm{MP}}^{*(m)}=\mathbf{O}_{\mathrm{MP}}^{(m)} \mathbf{H}^{(m)} \in \mathbb{R}^{d^{\prime \prime} \times J},
$$

其中，${\bf O}_\text{MP}^{(m)}\!\in\! \mathbb{R}^{d''\times N_{E_m}}$ 是从 ${\bf O}_\text{MP}$ 中简单地提取出阶数为 $m$ 的超边的特征表示，矩阵 ${\bf H}^{(m)}\!\in\! \mathbb{R}^{N_{E_m}\times J}$ 是可学习的权重，用于对阶数为 $m$ 的超边进行加权池化。最后，通过简单地连接 ${\bf O}_\text{MP}^{*(1)},\cdots,{\bf O}_\text{MP}^{*(r)}$，我们得到 ${\bf O}_\text{MP}^{*}\!\in\! \mathbb{R}^{r{d''\times J}}$。如果我们使用了从 TP 到 MP 的输入，则将 MP 的输出表示为 ${\mathbf{O}'}_\text{MP}^{*}$。

### Temporal block Pooling (TP) Module
**CmSA in TP**：首先，我们将多阶特征表示 $\vec{\mathcal{M}}\!\in\! \mathbb{R}^{d'\!\times\!N\!\times\!\tau}$ 重新整形为 ${\bf M}\!\in\! \mathbb{R}^{d'N\!\times\!\tau}$（或者将来自 MP 的输出重新整形为 ${\bf M}''\!\in\! \mathbb{R}^{rd'J\!\times\!\tau}$）。为简单起见，我们在第一种情况下记 $d'''\!=\!d'N$，在第二种情况下记 $d'''\!=\!rd'J$。在第一种情况下，重新整形后的输入的第一模式用于形成令牌，它们再次是耦合模式令牌，例如“通道-超边”和“阶-通道-身体关节”令牌，分别对应不同的表示意义。
此外，TP（可能是指某种处理方式或模块）还沿着块-时间模式（沿 $\tau$ 方向）执行池化操作。我们形成一个耦合模式自注意力：
$$
\begin{equation}
    a_\text{TP}({\bf Q}_\text{TP}, {\bf K}_\text{TP}, {\bf V}_\text{TP}) =\text{SoftMax}\left(\frac{\mathbf{Q}_\text{TP}\mathbf{K}_\text{TP}^\top}{\sqrt{d_{K_\text{TP}}}}\right)\mathbf{V}_\text{TP},
\end{equation}
$$
这里，$\sqrt{d_{K_\text{TP}}}$ 是缩放因子，${\bf Q}_\text{TP} = {\bf W}_\text{TP}^q{\bf M}$，${\bf K}_\text{TP} = {\bf W}_\text{TP}^k{\bf M}$ 和 ${\bf V}_\text{TP} = {\bf W}_\text{TP}^v{\bf M}$（我们可以使用 ${\bf M}$ 或者 ${\bf M}''$）分别是查询、键和值。此外，${\bf Q}_\text{TP}$，${\bf K}_\text{TP}$，${\bf V}_\text{TP} \in  \mathbb{R}^{d'''\times \tau}$ （或者 $\mathbb{R}^{3d'J\!\times\!\tau}$） 和 ${\bf W}_\text{TP}^q$，${\bf W}_\text{TP}^k$，${\bf W}_\text{TP}^v \in  \mathbb{R}^{d'''\times d'''}$ （或者 $\mathbb{R}^{3d'J\times 3d'J}$）是可学习的权重。方程式(10)重新加权 ${\bf V}_\text{TP}$，其基于联合模式令牌（例如“通道-超边”或“阶-通道-身体关节”）的 ${\bf Q}_\text{TP}$ 和 ${\bf K}_\text{TP}$ 令牌嵌入之间的相关性。注意力的输出是时间表示 ${\bf O}_\text{TP}  \in   \mathbb{R}^{d'''\times \tau}$。如果我们使用 ${\bf M}''$ 作为输入，则将输出表示为 ${\bf O}''_\text{TP}$。

**Pooling step**:在给定时间表示 ${\bf O}_\text{TP}\!\in\!\mathbb{R}^{d'''\!\times\!\tau}$（或者 ${\bf O}''_\text{TP}$）后，我们在块-时间模式（即 $\tau$ 方向）上应用池化操作，以获得与骨骼序列长度（块数量 $\tau$）无关的紧凑特征表示。有许多池化操作(我们没有提出池化算子，而是选择了一些流行的算子，以比较它们对 TP 的影响)。，包括一阶的（例如平均池化、最大池化、求和池化）、二阶的（如注意力池化）、高阶的（三线性池化） 和排序池化。

池化后的输出是 ${\bf O}^*_\text{TP}\!\in\!\mathbb{R}^{d'''}$（或者 ${{\bf O}''}^*_\text{TP}$）。 （或者 ${\bf O}^{*'}_\text{TP}\!\in\!\mathbb{R}^{d'N}$）