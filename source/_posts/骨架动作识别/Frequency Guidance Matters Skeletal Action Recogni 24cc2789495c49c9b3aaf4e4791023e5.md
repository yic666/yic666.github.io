---
title: Frequency Guidance Matters: Skeletal Action Recognition by Frequency-Aware Mixed Transformer
tags:
  - 骨架
  - 动作识别
categories:
  - 骨架动作识别
date: 2024-08-01
---

# 摘要

最近，Transformer在从骨骼序列中建模长期依赖性方面展示了巨大的潜力，因此在骨骼动作识别领域获得了越来越多的关注。然而，现有的基于Transformer的方法主要依赖于朴素注意力机制来捕捉时空特征，这在学习表现出相似运动模式的判别性表示方面存在不足。为了解决这一挑战，我们引入了**频**率感知**混**合**变**压器（FreqMixFormer），专门设计用于识别具有细微判别性运动的相似骨骼动作。首先，我们引入了一个频率感知注意力模块，通过将关节特征嵌入到频率注意力图中，来解开骨骼频率表示，旨在基于其频率系数区分判别性运动。随后，我们开发了一个混合Transformer架构，将空间特征与频率特征结合，以建模全面的频率-空间模式。此外，还提出了一个时间Transformer来提取跨帧的全局相关性。大量实验表明，FreqMixFormer在三个流行的骨骼动作识别数据集上优于SOTA，包括NTU RGB+D、NTU RGB+D 120和NW-UCLA数据集。我们的项目已在以下公开链接中提供[https://github.com/wenhanwu95/FreqMixFormer](https://github.com/wenhanwu95/FreqMixFormer)。

![Untitled](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/Untitled.png)

<!--more-->


# 引言

作者认为基于Transformer的方法与基于GCN的方法的差距在于：

1. **GCNs的优势**：图卷积网络（GCNs）通过局部化的图卷积操作，有效地捕捉人体的空间配置，这对于动作识别是必不可少的。
2. **Transformer的局限性**：传统的Transformer方法使用全局自注意力操作，利用全局自关注操作，缺乏固有的**归纳偏置**来掌握骨架的拓扑结构。
    1. **局部交互的稀释**：Transformer的全局自注意力机制可能会削弱关节间的微妙局部交互，这些交互对于区分相似动作是重要的。
    2. **注意力标准化问题**：在Transformer中，由于注意力分数是在整个序列上进行**标准化**的，那些对整体注意力格局影响不大的微妙但关键的区分特征可能会被忽略。

Contribution：

1. We propose a Frequency-aware Attention Block(FAB) to investigate frequency features within skeletal sequences. A frequency operator is specifically designed to improve the learning of frequency coefficients, thereby enhancing the ability to capture discriminative correlations among joints.
2. Consequently, we introduce the Frequency-aware Mixed Transformer (FreqMixFormer) to extract frequency-spatial joint correlations. The model incorporates a temporal transformer designed to enhance its ability to capture temporal features across frames.
3. Our proposed FreqMixFormer outperforms state-of-the-art performance on three benchmarks, including NTU RGB+D , NTU RGB+D 120, and Northwestern-UCLA.

# 方法

### Overview

![Untitled](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/Untitled%201.png)

## 用于关节序列编码的离散余弦变换（DCT）

输入：令$ x \in \mathbb{R}^{J \times C \times F }$表示输入的关节序列，第 $j$ 个关节在 $F$ 帧上的轨迹表示为$X_j = (x_{j,0}, x_{j, 1}, ... , x_{j, F})$

用DCT去学习序列的频率表示，而且保留所有DCT系数，还增强高频部分并降低低频部分，由于：

1.  高频DCT分量对空间域中难以区分的细微差异更加敏感（例如，图中所示的读写时的 手部动作）
2. 低频DCT系数反映具有稳定或静态运动模式的动作，这些动作在识别中不够具有区分性（例如，图 中所示的读写时的下半身动作）
3. 余弦变换具有出色的能量压缩特性，能够将大部分能量（低频系数）集中在变换的前几个系数中，适合放大微妙的运动特征。

计算：

对于关节轨迹 $X_j$，第$i$个DCT系数计算公式为
$$
C_{j,i} = \sqrt{\frac{2}{F}} \sum_{f=1}^{F} x_{j,f} \frac{1}{\sqrt{1 + \delta_{i1}}} \cos\left[\frac{\pi(2f - 1)(i - 1)}{2F}\right] 
$$


- 其中$ \delta_{ij} = 1$ 当 $i = j$ 时，否则 $ \delta_{ij} = 0$。 特别地，$i \in {1, 2, \ldots, F }$，$i$ 越大，频率系数越高。

使用逆离散余弦变换 (IDCT) 恢复时域中的原始输入序列，其公式如下：
$$
x_{j,f} = \sqrt{\frac{2}{F}} \sum_{i=1}^{F} \ C_{j,i} \frac{1}{\sqrt{1 + \delta_{i1}}} \cos\left[\frac{\pi(2f - 1)(i - 1)}{2F}\right] 
$$


##   Frequency-aware Mixed Transformer

![image-20240802111618007](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/image-20240802111618007.png)

### Mixed Spatial Attention

在分割输入 $ x_i \in \mathbb{R}^{J \times (C/n) \times F } $ 的基础上，沿空间维度提取每个序列的基本 Query 矩阵和 Key 矩阵：
$$
Q_i, K_i = ReLU (linear(AvgPool(x_i))),
$$
其中 $ i = 1,2, \ldots, n$。在这里，$ AvgPool $ 表示自适应平均池化，用于平滑关节权重并减少骨架数据中噪声或不相关变化的影响；一个带有 ReLU 激活操作的线性层（FC-layer）被应用以确保 $ Q_i$ 和 $ K_i$ 全局整合。然后，自注意力表达为：

$$
{Atten}^i_{self} = Softmax\left(\frac{Q_iK_i^T}{\sqrt{d}}\right) 
$$

为了在不同单元组之间实现更丰富的上下文整合，受 mixformer启发，提出了一种交叉注意力策略，其中 $ K_i$ 在相邻的注意力块之间共享。交叉注意力表达为：

$$
{Atten}^i_{mix} = Softmax\left(\frac{Q_{i+1}K_i^T}{\sqrt{d}}\right)
$$

每个混合注意力图表示为：

$$
MS_i = {Atten}^i_{self} + {Atten}^i_{mix} + {Atten}^{i-1}_{mix}
$$

这里的混合注意力图的数量基于单元组的数量（例如，Overview图中的 $ n = 3$）。这些混合注意力图由多个 SABs（Spatial Attention Blocks）提取，用于空间表示学习。

### Mixed Frequency-Spatial Attention

对分割后的关节序列 $ x_i $ 应用DCT，以获得相应的频率系数，并将其作为FABs（Frequency-aware Attention Blocks）的输入，表示为 $ DCT(x_i) $。与混合空间注意力类似，沿频率域获取 Query 和 Key 值：

$$
\overline{Q}_i, \overline{K}_i = ReLU (linear(AvgPool(DCT(x_i))))
$$

相应的基于频率的自注意力和混合注意力图表示为：

$$
\overline{Atten}^{i}_{self} = Softmax\left(\frac{{\overline{Q}_i \overline{K}_i^T}}{\sqrt{d}}\right)
$$

$$
\overline{Atten}^{i}_{mix} = Softmax\left(\frac{{\overline{Q}_{i+1} \overline{K}_i^T}}{\sqrt{d}}\right)
$$

因此，混合频率注意力图表示为：

$$
\overline{MF_i} = \overline{Atten}^i_{self} + \overline{Atten}^i_{mix} + \overline{Atten}^{i-1}_{mix}
$$

随后，采用频率操作符（FO）$\psi(\cdot)$对混合频率注意力图进行操作：$\psi (\overline{MF_i})$。给定频率操作符系数 $\varphi$，其中 $\varphi \in (0, 1)$，通过 $(1+\varphi)$ 增强 $\overline{MF_i}$ 中的高频系数，使微小和微妙的动作更加明显。另一方面，通过 $\varphi$ 减少低频系数，适当减少对显著动作的关注，同时保留整体动作表示的完整性。最佳 $\varphi$ 的搜索在第 \ref{sec:ablation} 节中讨论。之后，使用IDCT模块恢复变换后的骨架序列：$ MF_i = IDCT(\psi(\overline{MF_i})) $。所有的 $ M_i $ 由频率感知注意力块（FABs）提取，如图(b) 所示。

因此，输出为：$ MFS_i = MF_i + MS_i $，最终的混合频率-空间注意力图的输出可以表示为：

$$
M \leftarrow \text{Concat}[MFS_1, MFS_2, \ldots, MFS_i]
$$

我们通过在空间维度上添加一个空间 $ 1 \times 1 $ 卷积层，从初始输入 $ X $ 中统一计算获取 Value $ V $。因此，时间注意力块的输入表示为：

$$
x_t = MV
$$

### Temporal Attention Block

给定基于混合频率-空间注意力方法的时序输入 $ x_t $，采用一些策略来转换输入通道，并沿着时间维度获取更多的多变量信息：$ X_t = CT(x_t) $。

![image-20240802113204927](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/image-20240802113204927.png)

然后，将转换后的输入 $ X_t $ 通过时间注意力块（见图 (c)）处理，以获得相应的 Query 和 Key 矩阵：

$$
Q_t = \sigma(\text{linear}(\text{AvgPool}(X_t))), \\
K_t = \sigma(\text{linear}(\text{MaxPool}(X_t)))
$$

在这里，$ \sigma $ 表示激活函数，$ \text{AvgPool} $ 和 $ \text{MaxPool} $ 分别表示平均池化和最大池化。

时间注意力块中的 Value $ V_t $ 是通过对时序输入在时间维度上应用 $ 1 \times 1 $ 卷积层后从时序输入中获得的。最后，时间注意力表达为：

$$
Atten_{tem} = Softmax\left(\frac{Q_t K_t^T}{\sqrt{d}}\right),
$$

并且分类头的最终输出定义为：

$$
X_{out} = (\text{Sigmod}(Atten_{tem})) V_t
$$

在这里，$ \text{Sigmod} $ 表示 Sigmoid 激活函数。



# 实验

![image-20240802113302518](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/image-20240802113302518.png)

![image-20240802113318618](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/image-20240802113318618.png)

![image-20240802113328061](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/image-20240802113328061.png)

![image-20240802114449137](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/image-20240802114449137.png)

![image-20240802115910648](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/image-20240802115910648.png)

![image-20240802120203670](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/image-20240802120203670.png)

![image-20240802123403819](https://yic-123.oss-cn-guangzhou.aliyuncs.com/img/image-20240802123403819.png) 

上图展示了 FreqMixFormer 学习到的注意力矩阵的可视化结果。 骨架配置由 NTU-60 数据集生成（图  (a)）。 以“吃饭”为例（图(b)）。在相关性矩阵中，黄色越饱和表示权重越大，表明关节之间的相关性越强。数字表示不同的关节。
需要注意的是，混合空间注意力图（图  (c)，由 SAB 学习得到）表示关节之间的空间关系。 混合频率注意力图（图  (d)，由 FAB 学习得到）表明了运动的频率方面。 基于这两个注意力图，提出了混合频率-空间注意力图（图  (e)），用于同时捕获空间相关性和频率依赖性，从而整合空间和频率骨架特征。

从图中可以看出，模型在空间域中关注的是与脊柱和右手尖的相关性。而在频域中，模型关注了更多相关性区域（与脊柱、左臂的关节连接，以及头部和手之间的交互），这表明模型正在分析在空间域中被忽视的、更具区分性的运动。同时，混合频率-空间注意力图不仅包含从空间空间学习到的强注意力区域，还包含在频率空间中关注的相关性。