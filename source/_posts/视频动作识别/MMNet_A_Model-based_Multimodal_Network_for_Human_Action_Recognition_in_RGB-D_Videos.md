---
title: >-
  《MMNet: A Model-based Multimodal Network for Human Action Recognition in RGB-D
  Videos》阅读笔记
tags:
  - 动作识别
  - 论文笔记
  - RGB-D Action Recognition
categories:
  - 视频动作识别
date: 2023-01-08 10:00:59
---
2022 tpami

## 摘要

自廉价深度传感器问世以来，RGB-D视频中的人体动作识别(HAR)得到了广泛研究。目前，单模态方法(如基于骨架和基于RGB视频)已经在越来越大的数据集上实现了实质性的改进。然而，很少研究具有模型级融合的多模态方法。本文提出一种基于模型的多模态网络(MMNet)，通过一种基于模型的方法融合骨架和RGB模态。该方法的目标是通过有效地利用不同数据模态的互补信息来提高集成识别的精度。对于基于模型的融合方案，我们对骨架模态使用时空图卷积网络来学习注意力权重，并将其迁移到RGB模态的网络中。在5个基准数据集上进行了广泛的实验:NTU RGB+D 60、NTU RGB+D 120、PKU-MMD、Northwestern-UCLA Multiview和Toyota smarhome。在聚合多个模态的结果后，发现所提出方法在五个数据集的六个评估协议上优于最先进的方法;因此，MMNet能够有效地捕获不同RGB-D视频模态中相互补充的特征，为HAR提供更具判别力的特征。在包含更多户外动作的RGB视频数据集Kinetics 400上测试了MMNet，结果与RGB- d视频数据集的结果一致。

<!--more-->

## 序言

使用骨架或RGB模态的单模态方法存在障碍:

1. 基于RGB的方法的主要限制是缺乏3D结构；

2. 基于骨骼的方法也受到缺乏纹理和外观特征的限制。

    

![](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/image-20230105201002470.png)

多模态HAR方法的核心任务是数据融合，可进一步分为数据级融合、特征级融合和决策级融合

除了这三种之外还有一种融合方法，称为协同学习

为了推进之前围绕协作学习的工作，本文提出了一种新的基于模型的多模态网络(MMNet)，在融合骨架和RGB模态时建模有效的知识转换，以提高RGB- D视频中的人体动作识别

通过构建感兴趣的时空区域(STROI)特征图来关注整个RGB视频帧的不同外观特征，这种策略减轻了与大量视频数据相关的挑战。

从所提出的MMNet的骨架关节流Skeleton Joints中衍生出一个注意力掩码，以关注提供互补特征的ST-ROI区域，可以提高RGB-D视频中人体动作的识别。

贡献：

1. 首先，引入了一种多模态深度学习架构，在模型层次上用注意力机制融合不同的数据模态，并使用骨架骨流Skeleton Bones。
2. 其次，通过三个基准数据集证明，所提出方法大大提高了最先进的性能
3. 通过对MMNet的两个关键参数进行分析，进一步验证了该方法的有效性。

## 网络架构

![网络架构](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/image-20230106140433610.png)

$B^{(i)}$,$J^{(i)}$和$V^{(i)}$分别代表骨骼、骨骼关节和RGB视频的输入;

$w^{(i)}$是来自骨骼关节图表示的空间注意力权重$\hat{J}^{(i)}$，它指导从RGB视频输入$V^{(i)}$转换为ST-ROI的焦点;

在这种基于模型的数据融合之后,以骨架为中心的ST-ROI$R^{\prime(i)}$将被馈送到ResNet以生成特定模式的预测;

$\hat{y}_{c}^{J^{(i)}}$和$\hat{y}_{c}^{B^{(i)}}$分别表示来自skeleton joint 和 bone流的预测,这些预测通过RGB模态的特定预测$\hat{y}_{c}^{V^{(i)}}$进行聚合，以提供集成识别结果。

### 从RGB模态构建ST-ROI

基于视频的模比如I3D和S3D等需要大量的RAM和GPU显存的计算资源，并且需要很长时间才能收敛,而较早的模型如C3D在NTU RGB+D上则受限于数据量并不能有好的表现.因此,作者建议从RGB模态构建ST-ROI，并使用通用CNN模型从中检索有效特征。

以符号$V=\left\{V^{(i)} \mid i=1, \ldots, N\right\}$表示为有N个视频样本进行训练的RGB模态,那么可以表示出$V^{(i)}=\left(f_{1}^{(i)}, \ldots, f_{t}^{(i)}, \ldots, f_{T}^{(i)}\right)$

其中$f_{t}^{(i)}$是$t$帧。给定一个RGB帧$f_{t}^{(i)}$，作者定义了一个函数$g$来构建空间ROI$R_{t j}^{(i)}$为
$$
R_{t j}^{(i)}=g\left(f_{t}^{(i)}, o_{t j}^{(i)}\right), j \in\left(m_{1}, \ldots, m_{M_{O}^{\prime}}\right), M_{O}^{\prime} \leq M_{O}
$$
其中$O_{t j}^{(i)}$为时刻t时OpenPose骨架的第j个关节。

![构建ST-ROI](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/image-20230106193800339.png)

如上图，在$V^{(i)}$进行时间采样，选择L个代表帧，将它们拼接成一个方形ST-ROI，如上图中的单受试者案例所示。对于有两个主体的动作，我们裁剪每个主体的ST-ROI，如上图中两个主体的情况所示。ST-ROI显著减少了RGB视频输入的数据量，同时保留了物体的外观和动作的运动信息。在$τ$时刻的时域子ROI将具有$M′$个空间子ROI,可以垂直连接并表示为$R_{\tau}^{(i)}$;相反，第$j$个关节的空间子ROI将具有$L$个时间子ROI，可以水平级联并表示为$R^{(i)}_ j$;最后对于$V^{(i)}$的ST-ROI可用$R_{(i)}$表示,包含$M'\times L$个子ST-ROI$R_{\tau j}^{(i)}$

 

### 从骨架模态中学习关节权重

![时空骨架图结构和图卷积网络的空间采样策略](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/image-20230106204310906.png)

(a)时空骨架图结构。(b)图卷积网络的空间采样策略。不同的颜色表示不同的子集:绿色星形表示顶点本身;黄色三角形表示离心力较远的子集;蓝色方块表示更接近的向心子集。

符号表示:

$J^{(i)}=\left(J_{1}^{(i)}, \ldots, J_{t}^{(i)}, \ldots, J_{T}^{(i)}\right)$表示从时间$t=1$开始到时间$T$结束的第$i$个训练样本对应的$T$个骨架帧序列；

$B^{(i)}=\left(B_{1}^{(i)}, \ldots, B_{t}^{(i)}, \ldots, B_{T}^{(i)}\right)$表示由骨骼关节转化而来的骨骼的相应序列；

$J_{t}^{(i)}=\left(J_{t 1}^{(i)}, \ldots, J_{t j}^{(i)}, \ldots, J_{t M}^{(i)}\right)$表示给定一组t时刻观察到的骨架中的M个关节得到的骨架数据;

构造一个时空图来表示$J^{(i)}$的时空结构,即上图(a),其中单个骨架框架的关节和骨骼分别由图顶点([a]中的橙色圆圈)及其自然连接([a]中的紫色线)表示。两个相邻的骨架通过关节之间的边连接起来([a]中的黑线虚线)。图顶点的属性可以是每个关节对应的三维坐标。骨架输入$J^{(i)}$的骨架图可以符号化为$\mathcal{G}=(\mathcal{V}, \mathcal{E})$，其中$\mathcal{V}$和$\mathcal{E}$分别表示关节和骨骼。

#### 图卷积运算

为表示卷积操作的采样区域，一个节点$v_{ti}$的邻居集被定义为$N\left(v_{t i}\right)=\left\{v_{t j} \mid d\left(v_{t i}, v_{t j}\right) \leq D\right\}$,其中D是$d\left(v_{t i}, v_{t j}\right)$的最大路径长度.该策略如图4(b)所示，其中×表示骨架的重心,采样面积$N\left(v_{t i}\right)$被曲线包围.

假设在邻居集中有固定数量的K个子集,它们将被映射为数字标记$l_{t i}: N\left(v_{t i}\right) \rightarrow\{0, \ldots, K-1\}$;在时间上，邻域概念扩展到时间连接的关节，如$N\left(v_{t i}\right)=\left\{v_{q j}\left|d\left(v_{t j}, v_{t i}\right) \leq K,\right| q-t \mid \leq \Gamma / 2\right\}$,其中$\Gamma$是控制邻居集的时间范围的时间内核大小。这样图卷积就可以计算为
$$
\hat{v}_{t i}=\sum_{v_{t j} \in N\left(v_{t i}\right)} \frac{1}{Z_{t i}\left(v_{t j}\right)} f_{i n}\left(v_{t j}\right) \mathbf{w}\left(l\left(v_{t j}\right)\right)
$$
其中$f_{i n}\left(v_{t j}\right)$为获取$v_{tj}$的属性向量的特征映射，$\mathbf{w}\left(l\left(v_{t j}\right)\right)$是权重函数$\mathbf{w}\left(v_{t i}, v_{t j}\right)$:$N\left(v_{t i}\right) \rightarrow \mathbb{R}^{C}$可以用$(C,K)$维张量实现;$Z_{t i}\left(v_{t j}\right)=\left|v_{t k}\right| l_{t i}\left(v_{t k}\right)=l_{t i}\left(v_{t j}\right) \mid$等于相应子集的基数，这是一个归一化项。

#### 关节权重

对骨架模态应用图卷积后，图上每个顶点的输出可以用来推断相应骨架节点的重要性。骨架序列的特征映射可以用(C, T, M)维的张量表示，其中C表示关节顶点的属性个数，T表示时间长度，M表示顶点个数。这种划分策略可以用一个邻接矩阵A来表示，矩阵A中的元素表示一个顶点$v_{ti}$是否属于$N(v_{ti})$的子集。因此，图卷积可以使用$1 \times \Gamma$经典二维卷积，并通过在二维上将所得张量乘以归一化邻接矩阵$\boldsymbol{\Lambda}^{-\frac{1}{2}} \mathbf{A} \boldsymbol{\Lambda}^{-\frac{1}{2}}$来实现。若采用K种分区策略$\sum_{k=1}^{K} \mathbf{A}_{k}$，图卷积的公式可被转换为
$$
\hat{J}^{(i)}=\sum_{k=1}^{K} \boldsymbol{\Lambda}^{-\frac{1}{2}} \mathbf{A} \boldsymbol{\Lambda}^{-\frac{1}{2}} f_{i n}\left(J^{(i)}\right) \mathbf{W}_{k} \odot \mathbf{M}_{k}
$$

其中$\boldsymbol{\Lambda}_{k}^{i i}=\sum_{j}\left(\mathbf{A}_{k}^{i j}\right)+\alpha$为对角矩阵且$\alpha$被设为0.001以避免空行；$\mathbf{W}_{k}$是一个具有$(C_in, C_out, 1,1)$维的1 x 1卷积运算的权重张量，它表示方程3的权重函数；$\mathbf{M}_{k}$是与$A_k$相同大小的注意力图，表明了每个顶点的重要性;$\odot$表示两个矩阵的元素乘积；$\hat{J}^{(i)}$是一个大小为$(c, t, M)$的张量，其中$c$是输出通道数，$t$是输出时间长度，$M$是顶点数。该张量可用于推断动作类别，并可转换为关节权重，为RGB模态提供注意力知识。代表其相应身体面积重要性的关节权重可以计算为
$$
w^{(i)}=\frac{1}{c t} \sum_{1}^{c} \sum_{1}^{t} \sqrt{\left(\hat{J}_{c t}^{(i)}\right)^{2}}
$$
其中t和c分别为卷积图的输出维数，分别表示时间长度和输出通道。$w^{(i)}$是包含M个不同骨架关节权重的向量。

### 基于模型的融合

本文提出一种RGB帧的空间权重机制，使机器能够关注将提供判别式信息的RGB特征，更明确的是，机器将更有能力，因为它直观地模仿了人眼的动作识别。本文选择使用来自骨架模态的关节权重，并将其乘以ST-ROI来正则化RGB模态。可以将第i个训练样本的骨架聚焦的ST-ROI(记为$R^{'(i)}$)从$R^{(i)}$映射出来，函数$h$定义为
$$
R^{\prime(i)}=h\left(R_{j}^{(i)}, w_{j}^{(i)}\right), j=m_{1}^{\prime}, \ldots, m_{M^{\prime}}^{\prime}, M^{\prime}<M
$$
其中$w_{j}$为第$j$个关节的权重，$R_{j}^{(i)}$为对应身体区域的子空间ROI。而$m_{1}^{\prime}, \ldots, m_{M^{\prime}}^{\prime}$是$M’$个不同骨骼关节对应于建议关注的身体区域的指数。$M '$的值等于公式2中$M ' _O$的值。公式6的数据融合过程如下图所示。

![Model-based fusion scheme](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/image-20230107180156852.png)

### 目标函数

用动作标签监督的损失项集合的和构建了MMNet的端到端格式，表示为
$$
\mathcal{L}=\mathcal{L}_{J}\left(\hat{y}^{J}, y\right)+\mathcal{L}_{B}\left(\hat{y}^{B}, y\right)+\mathcal{L}_{V}\left(\hat{y}^{V}, y\right)
$$
其中$\mathcal{L}_{J},\mathcal{L}_{B},\mathcal{L}_{V}$分别为骨架关节、骨架骨骼、RGB视频输入的损失项。

将骨架关节输入输入到公式4中引入的图卷积模型中。因此，骨骼关节的预测可以定义为
$$
\hat{y}^{J^{(i)}}=\sigma\left(G_{J}\left(\Theta_{J}, J^{(i)}\right)\right)
$$
其中$G_J$表示式4定义的图卷积运算；$Θ_J$为GCN子模型的可学习参数；$J^{(i)}$是骨骼关节输入的数据样本。而$σ$表示一个线性层，将子模型输出的形状转换为one - hot表示

骨骼输入基本上是骨骼关节输入的转换。将同样的图卷积运算方法应用于骨骼输入，可以表示为
$$
\hat{y}^{B^{(i)}}=\sigma\left(G_{B}\left(\Theta_{B}, B^{(i)}\right)\right)
$$
本文提出了ST-ROI作为RGB视频输入的转换形式，它可以大幅减少数据量，并保持HAR的核心判别信息。由于ST-ROI本质上是一个二维特征图，便采用ResNet如下
$$
\hat{y}^{V^{(i)}}=\sigma\left(G_{V}\left(R^{\prime(i)}, \Theta_{V}\right)+R^{\prime(i)}\right)
$$
其中$G_{V}\left(R^{\prime(i)}, \Theta_{V}\right)$表示待学习的残差映射，ΘV表示基于ResNet层数的可学习参数

给定上述子模型预测的定义，根据以下目标制定优化问题:
$$
\begin{array}{c}
\underset{\Theta_{B}}{\arg \min }-\sum_{i=1}^{N} \sum_{c=1}^{N_{c}} \underbrace{y_{c} \log \left(\hat{y}_{c}^{B^{(i)}}\right)}_{\mathcal{L}_{B}} \\
\underset{\Theta_{J}}{\arg \min }-\sum_{i=1}^{N} \sum_{c=1}^{N_{c}} \underbrace{y_{c} \log \left(\hat{y}_{c}^{J^{(i)}}\right)}_{\mathcal{L}_{J}} \\
\underset{\Theta_{V}}{\arg \min }-\sum_{i=1}^{N} \sum_{c=1}^{N_{c}} \underbrace{y_{c} \log \left(\hat{y}_{c}^{V^{(i)}}\right)}_{\mathcal{L}_{V}}
\end{array}
$$
其中$\mathcal{L}$是交叉熵损失函数，$N_c$是特定数据集中动作类的数量，$N$表示训练集中的样本个数。

### 训练与优化

为了追求更高的识别精度，还可以采用其他几个损失项作为关节权重。但本文依旧采用了关节权重的普通实现，作为RGB模态的空间注意力，以验证新颖的基于模型的数据融合机制的有效性。给定目标函数，使用随机梯度下降(SGD)求解方程11、12和13。网络$G_J$可以预训练，也可以与$G_V$同时训练，以获得空间注意力权重以进行特征融合。子模型$G_J$和$G_V$可以通过将$Θ_J$和$Θ_V$一起调优来进行端到端训练，或者简单地通过修改$Θ_J$来更新$Θ_V$。同时，对骨骼的网络$G_B$进行单独训练，并将其聚合到$G_J$和$G_V$的结果中，从而实现集成预测。具体训练步骤如算法1所示。

![算法1](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/image-20230107205547516.png)