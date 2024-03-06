---
title: PoseConv3d笔记
tags:
  - 动作识别
  - 论文笔记
  - 骨架
categories:
  - 骨架动作识别
date: 2022-09-22 16:48:46
---

[paper](https://arxiv.org/abs/2104.13586)[code](https://github.com/kennymckormick/pyskl)

# 摘要与结论

- 尽管许多基于GCN的骨架动作识别算法取得不错的结果，但依旧在鲁棒性、互操作性和可扩展性方面存在限制。
- 提出了PoseConv3D：一种以3D热图体积作为输入的基于3D-CNN的骨骼动作识别方法，与GCN的方法相比
  - 在学习时空特征方面更加有效
  - 对姿态估计的噪声更具有鲁棒性
  - 在交叉数据集中更具有泛化性
  - 在处理多人场景方面无需额外计算成本

- 另外，更容易与其他模态结合，在八个多模态识别基准达到了SOTA

<!--more-->

# 引言

基于人体骨架的动作识别其动作聚焦性和紧凑性，近年来受到越来越多的关注。在实践中，视频中的人体骨架主要表示为一系列的关节坐标列表，其中的坐标由姿态估计器提取。GCN是最受欢迎的方法之一，具体地说，GCN将每个时间步长的每个人体关节视为一个节点，空间和时间维度上的相邻节点通过边连接起来，然后将图卷积层应用于所构建的图，以发现跨空间和时间的动作模式。

基于GCN的方法在以下方面有局限性:

- 鲁棒性：由于GCN直接处理关节坐标，坐标上的微小扰动通常导致完全不同的预测。
- 互操作性：由于GCN是在骨架图上操作的，因此难以与其他模态结合。
- 可扩展性：由于GCN将每个人体关节视为节点，因此涉及多人的场景中复杂性线性上升。

本文提出了PoseConv3D，解决了GCN方法的局限性：

- 使用3D热图表示骨架对姿态估计更具有鲁棒性，对不同方法获得的输入挂架具有很好的泛化能力
- 依赖于热图表示，更容易与其他模态集成到多流网络
- 热图表示的复杂度与人数无关，处理多人场景不会增加计算开销

# 网络结构

## 姿态提取的良好实践

人体骨骼或姿态提取是基于骨骼的动作识别的重要预处理步骤，对最终的识别精度有很大影响。

一般来说2D姿势比3D姿势效果更好，如下图。与自底向上的方法相比，自顶向下的方法在标准基准(如coco -关键点)上获得了优越的性能。

![](https://raw.githubusercontent.com/yic666/Blogimg/master/image-20220923200807607.png)

### 消融实验

作者设计了一系列采用的不同的替代方法的姿态提取的消融实验。以下的3D-CNN实验的输入均为$T*H*W=48*56*56$

#### 2D v.s. 3D 骨架

![](https://raw.githubusercontent.com/yic666/Blogimg/master/image-20220923201902224.png)

使用MS-G3D（用于基于骨骼的动作识别的当前最先进的GCN），对2D和3D关键点具有相同的配置和训练计划，结果如上表。

![](https://raw.githubusercontent.com/yic666/Blogimg/master/image-20220923204038409.png)

除了基于rgb的3d位姿估计方法，还考虑了“提升”方法，直接“提升”2d姿势(序列)到3d姿势(序列)，基于HRNet提取的2D姿态对3D姿态进行回归，利用提升后的3D姿态进行动作识别。上表的结果表明，这种被提升的3D姿势没有提供任何额外的信息，在动作识别方面的表现甚至比原始的2D姿势更差。

#### Bottom-Up v.s. Top-Down.

![](https://raw.githubusercontent.com/yic666/Blogimg/master/image-20220923210128369.png)

作者用相同的主干实例化这两种方法(HRNet-w32)。此外，作者还用MobileNet-v2骨干网实例化自顶向下方法进行比较，它在coco验证方面的性能与HRNet(自底向上)相似。上表的结果显示，HRNet(自下而上)在COCO-val上的性能远低于HRNet(自顶向下)，接近于MobileNet(自顶向下)。

#### Interested Person v.s. All Persons.

![](https://raw.githubusercontent.com/yic666/Blogimg/master/image-20220923210416469.png)

很多人可能存在于一个视频中，但并不是所有人都与感兴趣的动作有关。作者使用3种人物边界框进行姿态提取：Detection，Tracking（使用Siamese-RPN)和GT(对运动员的关注增加)。从上表的结果可以得到当事人的先验是极其重要的，即使是较弱的先验知识(每个视频1 个GT box)也能大大提高性能。

#### Coordinates v.s. Heatmaps

![](https://raw.githubusercontent.com/yic666/Blogimg/master/image-20220923211126851.png)

存储3D热图可能会占用大量磁盘空间。为提升效率，将每个 2D 关键点存储为坐标 (x, y, score)，其中 score 为预测的置信度。在 FineGYM 上进行了实验，以估计这种热图 → 坐标的压缩会带来多大信息损失。作者发现，在使用高质量特征提取器的情况下，使用坐标作为输入，动作识别的精度仅有少量下降 (0.4%)。因此在后续工作中，作者以坐标的格式来存储提取出的 2D 姿态。

## 从2D姿势生成3D热图

论文用大小为$K*H*W$的热图来表示二维姿势,其中K是关节点的数量，H和W是框架的高度和宽度。如果只有coordinate-triplets$(x_k; y_k; c_k)$,可以通过组合以每个关节为中心的K个高斯映射来得到一个关节热图J:

$$
\boldsymbol{J}_{k i j}=e^{-\frac{\left(i-x_{k}\right)^{2}+\left(j-y_{k}\right)^{2}}{2 * \sigma^{2}}} * c_{k}
$$

其中$\sigma$控制高斯映射的方差，$(x_k, y_k)$和$c_k$分别是第k个关节的位置和置信度分数，还可以创建limb热图：

$$
\boldsymbol{L}_{k i j}=e^{-\frac{\mathcal{D}\left((i, j), s e g\left[a_{k}, b_{k}\right]\right)^{2}}{2 * \sigma^{2}}} * \min \left(c_{a_{k}}, c_{b_{k}}\right)$$

第k个limb是在两个关节$a_k$和$b_k$之间。函数D计算从点$(i,J)$到段$\left[\left(x_{a_{k}}, y_{a_{k}}\right),\left(x_{b_{k}}, y_{b_{k}}\right)\right]$的距离。值得注意的是，尽管上述过程假设每一帧中都有一个人，但可以很容易地将其扩展到多人的情况，在这里直接累积所有人的第k个高斯映射，而无需放大热图。最后，一个3D热图堆叠是通过将所有热图($J$或$L$)沿时间维度堆叠而得到的，因此形状会是$K \times T \times H \times W$

实际应用中，作者使用了两种方法来尽可能减少 3D 热图堆叠中的冗余，使其更紧凑

1. Subjects-Centered Cropping

使热图与帧一样大是低效的，特别是当相关人员只在一个小区域活动时。在这种情况，先找到能够囊括了所有的2D姿势的边界框，然后根据找到的框裁剪所有帧，并将它们调整为目标大小。这样的话，2D姿势以及它们的移动能被保存，且使得三维热图体积的大小可以在空间上缩小。

2. Uniform Sampling.

通过对帧的子集进行采样，还可以沿时间维减小3D热图的体积。具体来说，为从视频中采样n帧，将视频分成n个等长的片段，并从每个片段中随机选择一帧。

![](https://raw.githubusercontent.com/yic666/Blogimg/master/image-20220924200056400.png)

## 用于基于骨架的动作识别的3D-CNN

### PoseConv3D

PoseConv3D以3D热图堆叠作为输入，可以用各种3D- cnn的backbone实例化。与一般的3D-CNN网络相比，需要添加两个修改：（1）由于3D热图体积的空间分辨率不需要像RGB剪辑那么大，因此在3D- cnn中删除了早期阶段的下采样操作；（2）由于采用的3D热图已经是中级特征，因此一个更浅(更少层)和更薄(更少通道)的网络对于PoseConv3D已经足够了。基于这些改动，作者采用了三种著名的3D-CNN：C3D，SlowOnly和X3D

![](https://raw.githubusercontent.com/yic666/Blogimg/master/image-20220924164508040.png)

如下表所示，采用轻量级版本的3d - cnn可以显著降低计算复杂度，但识别性能略有下降。而SlowOnly直接从Resnet膨胀而来而且具有良好的识别性能，作者将其作为Backbone。

![](https://raw.githubusercontent.com/yic666/Blogimg/master/image-20220924183705271.png)

### RGBPose-Conv3D

作者提出RGBPose-Conv3D用于早期的人体骨骼和RGB帧的融合，有两条通路分别处理RGB模态和Pose模态。总的来说，RGBPose-Conv3D的架构遵循几个原则：（1）相比于RGB流，Pose流具有较小的通道宽度和较小的深度，以及更小的输入空间分辨率；（2）加了Early Fusion，增加了两个通路之间的双向横向连接，促进两种模式之间的早期特征融合。RGBPose- Conv3D分别使用每个通路的两个单独损失进行训练，因为联合从两种模态学习的单个损失会导致严重的过拟合。

![](https://raw.githubusercontent.com/yic666/Blogimg/master/image-20220924195658960.png)

