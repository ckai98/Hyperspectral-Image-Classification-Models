# GCN/GNN 高光谱图像分类模型

本文档整理了项目中与 **图卷积网络 (GCN)** 和 **图神经网络 (GNN)** 相关的高光谱图像分类算法。

---

## 模型列表

| 模型名称 | 年份 | 路径 | 论文标题 |
|---------|------|------|---------|
| GCN_TGRS_DFH | 2020 | `2020/GCN_TGRS_DFH/` | Graph Convolutional Networks for Hyperspectral Image Classification |
| CNN_Enhanced_GCN | 2020 | `2020/CNN_Enhanced_GCN/` | CNN-Enhanced Graph Convolutional Network With Pixel- and Superpixel-Level Feature Fusion for Hyperspectral Image Classification |
| DRGCN | 2022 | `2022/DRGCN/` | Dual Residual Graph Convolutional Network for Hyperspectral Image Classification |
| F2HNN | 2022 | `2022/F2HNN/` | Hyperspectral Image Classification Using Feature Fusion Hypergraph Convolution Neural Network |
| MSSG-UNet | 2022 | `2022/MSSG-UNet/` | Multilevel Superpixel Structured Graph U-Nets for Hyperspectral Image Classification |
| AMGCFN | 2023 | `2023/AMGCFN/` | Attention Multihop Graph and Multiscale Convolutional Fusion Network for Hyperspectral Image Classification |
| SSGRN | 2023 | `2023/SSGRN/` | Spectral-Spatial Global Graph Reasoning for Hyperspectral Image Classification |
| MSA-GCN | 2024 | `2024/MSA-GCN/` | Interactive Enhanced Network Based on Multihead Self-Attention and Graph Convolution for Classification of Hyperspectral and LiDAR Data |
| SGMAE | 2025 | `2025/SGMAE/` | Self-Supervised Graph Masked Autoencoders for Hyperspectral Image Classification |

---

## 模型详细说明

### 1. GCN_TGRS_DFH (2020)
- **类型**: 基础 GCN
- **论文**: Graph Convolutional Networks for Hyperspectral Image Classification
- **特点**: 最基础的图卷积网络在高光谱图像分类中的应用，直接建模像素之间的图关系

### 2. CNN_Enhanced_GCN (2020)
- **类型**: CNN + GCN 混合
- **论文**: CNN-Enhanced Graph Convolutional Network With Pixel- and Superpixel-Level Feature Fusion for Hyperspectral Image Classification
- **特点**: 
  - 结合 CNN 和 GCN 的优势
  - 融合像素级和超像素级特征
  - 利用超像素降低计算复杂度

### 3. DRGCN (2022)
- **类型**: 双残差 GCN
- **论文**: Dual Residual Graph Convolutional Network for Hyperspectral Image Classification
- **特点**: 
  - 采用双残差结构
  - 有效缓解图卷积的过平滑问题
  - 增强深层网络的特征传播能力

### 4. F2HNN (2022)
- **类型**: 超图神经网络
- **论文**: Hyperspectral Image Classification Using Feature Fusion Hypergraph Convolution Neural Network
- **特点**: 
  - 使用超图 (Hypergraph) 卷积
  - 可以建模多节点之间的复杂关系
  - 超边可以连接多个节点，表达更丰富的结构信息

### 5. MSSG-UNet (2022)
- **类型**: Graph U-Net
- **论文**: Multilevel Superpixel Structured Graph U-Nets for Hyperspectral Image Classification
- **特点**: 
  - 将图结构与 U-Net 架构结合
  - 多级超像素结构化
  - 支持多尺度特征提取

### 6. AMGCFN (2023)
- **类型**: 注意力多跳图网络
- **论文**: Attention Multihop Graph and Multiscale Convolutional Fusion Network for Hyperspectral Image Classification
- **特点**: 
  - 注意力机制增强的多跳图卷积
  - 多尺度卷积特征融合
  - 有效捕获长距离依赖关系

### 7. SSGRN (2023)
- **类型**: 图推理网络
- **论文**: Spectral-Spatial Global Graph Reasoning for Hyperspectral Image Classification
- **特点**: 
  - 全局图推理机制
  - 联合建模光谱-空间信息
  - 捕获全局上下文关系

### 8. MSA-GCN (2024)
- **类型**: 多模态 Attention-GCN
- **论文**: Interactive Enhanced Network Based on Multihead Self-Attention and Graph Convolution for Classification of Hyperspectral and LiDAR Data
- **特点**: 
  - 多头自注意力与图卷积结合
  - 支持高光谱和 LiDAR 数据融合
  - 交互式特征增强

### 9. SGMAE (2025)
- **类型**: 自监督图掩码自编码器
- **论文**: Self-Supervised Graph Masked Autoencoders for Hyperspectral Image Classification
- **特点**: 
  - 最新的自监督学习方法
  - 图掩码自编码器架构
  - 无需大量标注数据即可学习有效表示

---

## 技术要点

### 为什么使用 GCN/GNN？
1. **非欧几里得数据建模**: 高光谱图像中像素之间的关系不限于规则的网格结构
2. **保留类别边界**: 相比 CNN，GCN 能更好地保留不同类别之间的边界信息
3. **长距离依赖**: 通过图结构可以建模远距离像素之间的关系
4. **降低计算成本**: 结合超像素分割可以显著减少计算量

### 常见技术组合
- **超像素 + GCN**: 使用超像素作为图节点，减少计算复杂度
- **CNN + GCN**: CNN 提取局部特征，GCN 建模全局关系
- **注意力机制 + GCN**: 动态学习图中边的权重
- **残差连接 + GCN**: 缓解深层图网络的过平滑问题

---

## 参考资料
- 各模型详细代码请参考对应年份文件夹
- 项目 README 和 AGENTS.md 包含更多使用说明
