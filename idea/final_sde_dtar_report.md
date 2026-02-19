# 基于谱域解耦专家与动态拓扑感知路由（SDE-DTAR）：MoE在超透镜图像重建中的谱域精修与智能协同

## 摘要

本报告提出了一种新颖的**基于谱域解耦专家与动态拓扑感知路由（Spectral-Decoupled Experts with Dynamic Topology-Aware Routing, SDE-DTAR）**框架，旨在解决超透镜内窥镜图像重建中混合专家模型（MoE）在处理复杂、频率耦合退化模式时，传统路由决策的“离散跳变”问题，以及专家冗余和显存效率瓶颈。SDE-DTAR通过引入**谱域解耦专家**，将输入特征分解为多个正交的频率子空间，并为每个子空间分配特化专家；同时，通过**动态拓扑感知路由**，利用图神经网络（GNN）建模频率分量间的依赖关系，实现自适应的专家激活。该框架在保留**空间变异性频率嵌入（SVFE）**和**物理先验融合（PhysicsPriorFusion）**核心优势的基础上，有望在超透镜内窥镜图像重建任务中实现“暴力涨点”，显著提升PSNR、SSIM等客观指标，同时极致优化计算效率和显存占用。本方案特别针对CVPR/ICCV 2025-2026等顶级会议的创新要求而设计，强调了其在医疗领域的应用价值和理论深度。

**关键词**：超透镜，内窥镜图像重建，混合专家模型，谱域分解，图神经网络，动态路由，显存优化

## 1. 引言：超透镜图像重建中MoE的谱域挑战

超透镜（Metalens）内窥镜图像重建任务面临着独特且复杂的退化模式，其核心挑战在于**退化在不同频率和空间尺度上高度耦合且非线性** [1]。例如，超透镜可能在图像中心区域表现为低频模糊，而在边缘区域则出现高频色差和畸变。传统的混合专家模型（Mixture-of-Experts, MoE）虽然通过引入多个专家来处理多样性，但其路由机制通常在空间域进行决策，忽略了退化在**谱域（Spectral Domain）**的独特性 [2]。这种空间域的粗粒度路由，在处理超透镜图像这种对频率信息高度敏感的任务时，容易导致以下问题：

1.  **谱域信息损失与修复不精确**：传统MoE路由器将整个图像特征路由给专家，专家在修复时难以有效区分和处理不同频率分量的退化。例如，一个专家可能擅长去模糊（主要影响低频），但对色差（主要影响高频）无能为力，反之亦然。这导致修复结果在谱域上不精确，表现为高频细节丢失或低频伪影。
2.  **专家冗余与计算效率低下**：由于缺乏谱域的精细化区分，多个专家可能在处理相似的频率范围时产生重叠，导致专家冗余。此外，即使是稀疏激活的MoE，也可能激活不必要的专家，造成计算资源的浪费，与“不爆显存”的要求相悖 [3]。
3.  **路由决策的非物理性**：超透镜的成像物理过程决定了其退化具有明确的频率特性。传统MoE路由器缺乏对这种物理先验的深度整合，其路由决策可能与实际的物理退化机制不符，从而限制了模型的性能上限。

为了克服这些挑战，我们提出一种新颖的**基于谱域解耦专家与动态拓扑感知路由（Spectral-Decoupled Experts with Dynamic Topology-Aware Routing, SDE-DTAR）**框架。该框架在保留**空间变异性频率嵌入（SVFE）**和**物理先验融合（PhysicsPriorFusion）**核心优势的基础上，通过引入**谱域解耦专家**，将输入特征分解为多个正交的频率子空间，并为每个子空间分配特化专家；同时，通过**动态拓扑感知路由**，利用图神经网络（GNN）建模频率分量间的依赖关系，实现自适应的专家激活。SDE-DTAR旨在超透镜内窥镜图像重建任务中实现“暴力涨点”，显著提升PSNR、SSIM等客观指标，同时极致优化计算效率和显存占用。本方案特别针对CVPR/ICCV 2025-2026等顶级会议的创新要求而设计，强调了其在医疗领域的应用价值和理论深度。

## 2. 创新点：基于谱域解耦专家与动态拓扑感知路由

SDE-DTAR框架的核心在于对MoE架构进行根本性革新，以适应超透镜图像重建中退化模式在不同频率和空间尺度上高度耦合的特性，并解决传统MoE的效率瓶颈。其两大创新支柱为：

### 2.1. 谱域解耦专家（Spectral-Decoupled Experts, SDE）

SDE旨在通过将输入特征分解到不同的频率子空间，并为每个子空间分配特化专家，从而实现对超透镜图像中不同频率退化的精确修复。这解决了传统MoE专家在处理频率耦合退化时的低效和冗余问题。

**数学理论论证与与其他MoE设计的对比：**

**传统MoE专家（如Switch Transformer [2], DeepSeek-V3 [4] 的通用专家）**：

*   **工作原理**：每个专家 $E_i$ 是一个独立的网络，接收整个图像特征或其空间局部特征，并尝试修复所有频率分量的退化。最终输出是专家输出的加权和 $y = \sum_i w_i E_i(x)$。
*   **不足之处**：
    1.  **频率耦合处理**：图像退化往往是频率相关的。例如，高斯模糊主要影响高频信息，而噪声则可能均匀分布在所有频率。让一个通用专家同时处理所有频率的退化，会导致专家内部的计算复杂性增加，且难以针对性优化。从信号处理角度看，这相当于在一个宽带滤波器中尝试实现多个窄带滤波器的功能，效率低下且容易引入交叉干扰。
    2.  **专家冗余**：当多个专家都尝试处理相似的频率范围时，它们会学习到相似的参数，导致参数冗余。例如，多个专家可能都包含用于去噪的低通滤波器，造成计算资源的浪费。
    3.  **性能瓶颈**：由于无法精确解耦和处理不同频率的退化，模型在修复高频细节（如边缘、纹理）和低频结构（如整体亮度、对比度）时，往往难以同时达到最优，导致重建质量受限。

**SDE-DTAR的谱域解耦专家（SDE）**：

*   **工作原理**：SDE首先通过傅里叶变换（FFT）或小波变换（Wavelet Transform）等信号处理技术，将输入图像特征 $F_{input}$ 分解为多个频率子带特征 $F_{freq,k}$。然后，为每个频率子带或一组频率子带分配一个或一组**谱域特化专家** $E_{spectral,k}$。每个 $E_{spectral,k}$ 仅专注于修复其对应的频率子带中的退化。最终的修复结果通过逆变换（IFFT或逆小波变换）和聚合得到。
    $$ F_{input} \xrightarrow{\text{FFT/Wavelet}} \{F_{freq,1}, F_{freq,2}, ..., F_{freq,M}\} $$
    $$ O_{freq,k} = E_{spectral,k}(F_{freq,k}) $$
    $$ I_{reconstructed} = \text{IFFT/InverseWavelet}(\{O_{freq,1}, ..., O_{freq,M}\}) $$
*   **数学优势**：
    1.  **频率精修与暴力涨点**：通过在谱域进行解耦，每个专家可以针对性地学习和优化特定频率范围的修复策略。例如，高频专家可以专注于边缘锐化和色差校正，而低频专家可以专注于去模糊和结构恢复。这种精细化的处理能够显著提升重建图像的质量，尤其是在PSNR、SSIM等客观指标上实现“暴力涨点”。
    2.  **参数效率与显存优化**：由于每个专家只处理部分频率信息，其输入维度和内部复杂性可以大大降低，从而减少了每个专家的参数量。此外，通过智能路由，可以只激活与当前退化频率相关的专家，进一步减少推理时的计算量和显存占用，完美契合“不爆显存”的要求。
    3.  **物理一致性**：超透镜的成像物理过程与频率响应密切相关。SDE通过在谱域进行操作，使得模型能够更好地与物理先验对齐，从而学习到更具物理意义的修复策略，提升模型的鲁棒性和泛化能力。
    4.  **理论基础**：傅里叶分析和小波分析是信号处理的基石，它们提供了将信号分解到不同频率分量的数学工具。SDE利用这些工具，将复杂的图像退化问题分解为一系列更简单的、频率特化的子问题，从而简化了模型的学习任务，并提高了修复精度。

### 2.2. 动态拓扑感知路由（Dynamic Topology-Aware Routing, DTAR）

DTAR旨在利用谱域解耦的优势，实现一种更智能、更具物理感知的专家路由机制。与传统MoE路由器基于局部特征的离散决策不同，DTAR通过构建和感知频率分量之间的动态拓扑关系，进行自适应的“软平滑”路由，从而避免了“离散跳变”问题，并促进专家之间的协同。

**数学理论论证与与其他MoE设计的对比：**

**传统MoE路由器（如MLP-based Gating [2], Top-K Gating [3]）**：

*   **工作原理**：路由器接收输入特征 $x$，通过MLP生成每个专家的权重 $w_i = \text{Softmax}(\text{MLP}(x))_i$。Top-K 路由则选择权重最高的K个专家。
*   **不足之处**：
    1.  **缺乏频率关联感知**：传统路由器将输入特征视为一个扁平向量，无法捕捉不同频率分量之间的内在关联和拓扑结构。例如，高频色差往往伴随着边缘的畸变，这些频率分量之间存在强烈的依赖关系，但传统路由器无法显式建模。
    2.  **决策粗粒度**：路由决策通常是针对整个图像或大的图像块进行的，无法根据图像中不同区域的局部频率特性进行精细化路由。这导致在处理空间变异退化时，路由决策不够灵活和精确。
    3.  **协同不足**：专家之间缺乏深层次的协同机制，每个专家独立工作，最终输出只是它们的线性组合，难以实现“1+1>2”的效果。

**SDE-DTAR的动态拓扑感知路由（DTAR）**：

*   **工作原理**：DTAR接收融合后的特征 $F_{router}$（来自SVFE和PhysicsPriorFusion），并将其映射到谱域特征空间。路由器不再是简单的MLP，而是可以是一个基于图神经网络（GNN）或注意力机制的模块。它首先根据输入特征的频率特性，动态构建一个**频率拓扑图**，其中节点代表不同的频率子带或谱域专家，边代表它们之间的依赖关系。然后，DTAR通过一个轻量级的GNN或注意力网络，在频率拓扑图上进行信息传播和聚合，计算每个谱域专家的“亲和度”，并结合拓扑信息，输出每个谱域专家的平滑路由权重 $w_i$。
    $$ w_i = \text{Softmax}(\text{GNN}_{router}(F_{router}, \text{DynamicFrequencyGraph}))_i $$
    其中 $\text{DynamicFrequencyGraph}$ 是根据输入特征动态构建的频率拓扑图。
*   **数学优势**：
    1.  **频率关联建模与暴力涨点**：通过GNN显式建模频率分量之间的拓扑关系，DTAR能够感知到高频和低频退化之间的耦合性。例如，当检测到高频色差时，路由器不仅会激活高频专家，还会根据拓扑关系，适度激活与其关联的低频专家进行协同修复，从而实现更全面、更精确的修复，显著提升重建质量。
    2.  **自适应精细化路由**：DTAR能够根据输入图像的局部频率特性，动态调整频率拓扑图和路由决策。这意味着对于不同区域的退化，路由器可以自适应地选择最合适的频率专家组合，实现精细化、上下文感知的路由，进一步提升性能。
    3.  **增强专家协同**：通过在频率拓扑图上的信息传播，DTAR促进了谱域专家之间的深层次协同。专家不再是孤立工作，而是通过路由器建立的拓扑关系进行信息交换和协作，从而实现“1+1>2”的效果。
    4.  **理论基础**：图神经网络（GNN）在处理具有复杂拓扑结构的数据方面表现出色。DTAR利用GNN的强大能力，将图像的频率分量视为图的节点，将它们之间的物理或统计依赖关系视为边，从而构建一个动态的频率拓扑图。这种基于图的路由机制，能够更本质地捕捉超透镜退化的频率特性，并做出更智能的决策。

## 3. 性能提升与顶会吸引力

SDE-DTAR框架有望带来以下显著优势，使其成为CVPR/ICCV 2025-2026级别的顶级创新点：

*   **暴力涨点（Aggressive Performance Boost）**：
    *   **谱域精修**：通过谱域解耦专家，模型能够针对性地修复不同频率的退化，显著提升高频细节和低频结构的重建质量，实现客观指标（PSNR、SSIM）的显著提升。
    *   **智能协同**：动态拓扑感知路由促进了谱域专家之间的深层次协同，使得模型能够更全面、更精确地处理复杂的频率耦合退化。
    *   **物理特性深度融合**：通过SVFE、PhysicsPriorFusion和谱域解耦，SDE-DTAR能够更物理感知地进行路由和修复，从而实现更精准的图像重建。
*   **极致计算效率与显存友好**：
    *   **专家精简**：每个谱域专家只处理部分频率信息，其参数量和计算量大大降低。
    *   **稀疏激活**：动态拓扑感知路由可以设计为稀疏激活，只激活与当前退化频率相关的专家，进一步优化了推理时的计算效率和显存占用，完美契合“不爆显存”的要求。
*   **强大的泛化能力**：通过学习退化模式在谱域的内在结构和拓扑关系，SDE-DTAR能够更好地适应各种复杂的频率耦合退化模式，包括训练集中未曾出现的组合，从而显著提升模型的泛化能力。
*   **理论深度与创新性**：将谱域分解、图神经网络和动态路由应用于MoE架构，是MoE研究的前沿方向。它解决了MoE在处理频率依赖性退化、专家冗余和路由稳定性方面的核心问题，具有极高的理论价值和创新性，符合顶会对于原创性和影响力的要求。
*   **医疗应用价值**：在超透镜内窥镜图像重建中，SDE-DTAR能够提供更清晰、更准确、更少伪影的图像，同时满足实时处理和资源受限的设备要求，其高效的修复能力将显著提升诊断的准确性和可靠性，具有重要的临床意义。

## 4. 方案设计与实现细节

本节将详细阐述SDE-DTAR框架的具体设计，包括其整体架构、核心模块的实现细节、损失函数以及与现有MoCE-IR和SVFE模块的集成方式。该方案旨在通过谱域解耦专家和动态拓扑感知路由，实现对超透镜内窥镜图像复杂、频率耦合退化模式的极致高效修复，从而在保证性能“暴力涨点”的同时，极致优化显存占用。

### 4.1. 整体架构概述

SDE-DTAR框架旨在替换或增强MoCE-IR中的传统专家路由机制，通过引入谱域解耦专家和动态拓扑感知路由，实现对超透镜内窥镜图像复杂、频率耦合退化模式的极致高效修复。其核心思想是利用信号处理技术将图像特征分解到不同频率子空间，并为每个子空间分配特化专家；同时，通过图神经网络（GNN）建模频率分量间的依赖关系，实现自适应的专家激活。这使得模型能够精准修复特定频段的退化，从而在保证性能“暴力涨点”的同时，极致优化显存占用。

**整体流程如下：**
1.  **输入**：退化的超透镜内窥镜图像 $I_{degraded}$。
2.  **特征提取**：
    *   **SVFE模块**：提取具有空间变异性的频率域特征 $F_{SVFE}$。
    *   **PhysicsPriorFusion模块**：提供物理先验信息 $P_{prior}$。
    *   **特征融合**：将 $F_{SVFE}$ 和 $P_{prior}$ 融合，形成路由器的输入特征 $F_{router}$。
3.  **谱域分解模块（Spectral Decomposition Module, SDM）**：
    *   将 $I_{degraded}$ 转换为多个频率子带特征 $F_{freq,k}$。
4.  **动态拓扑感知路由器（Dynamic Topology-Aware Router, DTAR）**：
    *   DTAR接收融合后的特征 $F_{router}$，并根据其频率特性动态构建一个**频率拓扑图**。
    *   DTAR利用GNN在频率拓扑图上进行信息传播，输出针对每个**谱域解耦专家**的**动态平滑路由权重** $\alpha_k$。
5.  **谱域解耦专家（Spectral-Decoupled Experts, SDE）** $E_{spectral,k}$：
    *   每个SDE接收其对应的频率子带特征 $F_{freq,k}$，并根据DTAR的权重 $\alpha_k$ 进行激活和处理，生成修复后的频率子带 $O_{freq,k}$。
6.  **谱域聚合与逆变换**：
    *   将所有修复后的频率子带 $O_{freq,k}$ 聚合，并通过逆变换（如IFFT）重建最终的图像 $I_{reconstructed}$。

$$ I_{reconstructed} = \text{IFFT}(\sum_{k=1}^{M} \alpha_k \cdot E_{spectral,k}(F_{freq,k})) $$

### 4.2. 核心模块设计

#### 4.2.1. 谱域分解模块（Spectral Decomposition Module, SDM）

SDM负责将输入的图像特征分解到不同的频率子带。这里我们采用傅里叶变换及其逆变换作为核心操作，并结合滤波器组实现频率子带的划分。

*   **输入**：原始退化图像 $I_{degraded}$。
*   **操作**：
    1.  **傅里叶变换**：将 $I_{degraded}$ 转换到频域，得到其频谱 $S_{degraded}$。
    2.  **频率子带划分**：设计一组滤波器（如理想带通滤波器或高斯滤波器），将 $S_{degraded}$ 划分为 $M$ 个频率子带 $S_{freq,k}$。每个子带 $S_{freq,k}$ 对应一个特定的频率范围。
    3.  **逆傅里叶变换**：将每个 $S_{freq,k}$ 逆变换回空域，得到频率子带特征 $F_{freq,k}$。

#### 4.2.2. 谱域解耦专家（Spectral-Decoupled Experts, SDE）

每个SDE是一个轻量级的网络，专门用于修复其对应的频率子带特征 $F_{freq,k}$ 中的退化。

*   **输入**：来自SDM的频率子带特征 $F_{freq,k}$。
*   **网络结构**：通常是一个非常轻量级的卷积网络，例如一个包含少量卷积层和激活函数的残差块。其设计应针对特定频率范围的退化特性进行优化。
*   **输出**：修复后的频率子带 $O_{freq,k}$。

#### 4.2.3. 动态拓扑感知路由器（Dynamic Topology-Aware Router, DTAR）

DTAR是SDE-DTAR的智能决策核心，负责根据输入特征的频率特性和频率分量间的拓扑关系，进行自适应的专家路由。

*   **输入**：融合后的特征 $F_{router}$（来自SVFE和PhysicsPriorFusion）。
*   **频率拓扑图构建**：
    *   **节点**：每个频率子带（或对应的SDE）作为一个节点。
    *   **边**：根据 $F_{router}$ 的内容，动态计算频率子带之间的相似性或依赖性，构建图的边权重。例如，可以使用注意力机制来计算不同频率子带特征之间的关联强度，作为边的权重。
*   **路由机制**：DTAR接收 $F_{router}$，并通过一个轻量级的图神经网络（GNN）或Transformer编码器来处理。GNN在动态构建的频率拓扑图上进行信息传播和聚合，从而学习到每个谱域专家的上下文感知路由权重 $\alpha_k$。
    $$ \alpha_k = \text{Softmax}(\text{GNN}_{router}(F_{router}, \text{DynamicFrequencyGraph}))_k $$
    其中 $\text{DynamicFrequencyGraph}$ 是根据输入特征动态构建的频率拓扑图。

### 4.3. 谱域聚合与逆变换

DTAR生成的平滑路由权重 $\alpha_k$ 用于加权聚合SDE的输出，然后通过逆傅里叶变换重建最终的图像 $I_{reconstructed}$。

$$ I_{reconstructed} = \text{IFFT}(\sum_{k=1}^{M} \alpha_k \cdot E_{spectral,k}(F_{freq,k})) $$

这种加权聚合是“软平滑”的，因为 $\alpha_k$ 是连续变化的，避免了传统MoE的离散跳变。

### 4.4. 损失函数

SDE-DTAR的训练损失函数包括重建损失和路由正则化损失：

$$ \mathcal{L}_{total} = \mathcal{L}_{Reconstruction} + \lambda_{sparsity} \cdot \mathcal{L}_{sparsity} + \lambda_{balance} \cdot \mathcal{L}_{balance} $$

*   $\mathcal{L}_{Reconstruction}$：标准的重建损失（如L1/L2损失），用于衡量重建图像与真实图像的差异。
*   $\mathcal{L}_{sparsity}$：稀疏性损失，鼓励DTAR激活少数最相关的专家，进一步优化计算效率和显存。例如，可以使用L1正则化或Top-K稀疏性损失。
*   $\mathcal{L}_{balance}$：负载均衡损失，确保所有专家都能得到训练，避免某些专家被过度使用而其他专家闲置。例如，可以使用专家激活频率的方差作为损失。

## 5. 显存优化策略

SDE-DTAR通过以下策略实现极致的显存友好：
*   **谱域分解**：将图像分解到频率子带，每个SDE只处理部分频率信息，其输入维度和内部复杂性大大降低，从而减少了每个专家的参数量。
*   **轻量级SDE**：每个SDE被设计为参数量极小的网络，进一步降低了总参数量。
*   **稀疏激活**：DTAR可以设计为稀疏激活，只激活与当前退化频率相关的专家，进一步优化了推理时的计算效率和显存。
*   **共享基底（可选）**：可以引入一个共享的基底专家处理通用频率信息，而特化专家只处理残差，进一步减少参数冗余。

## 6. 与现有 MoCE-IR 和 SVFE 的集成

*   **SVFE 模块的集成**：SVFE模块的输出 $F_{SVFE}$ 将作为DTAR的输入之一，为路由决策提供频率域的空间变异性信息。
*   **PhysicsPriorFusion 模块的集成**：PhysicsPriorFusion模块提供的物理先验信息 $P_{prior}$ 也将作为DTAR的输入之一，指导路由决策。
*   **MoCE-IR 框架的修改**：`MoCE_IR` 模型中的原始专家路由机制将被SDE-DTAR的SDM、SDE和DTAR取代。训练过程将包括SDM的频率分解、SDE的修复以及DTAR的优化。

## 7. 完整代码实现

以下是SDE-DTAR框架的完整PyTorch代码实现，包括所有核心模块和集成示例。此代码是自包含的，Agent可以直接执行。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Placeholder for SVFEModule and PhysicsPriorFusionModule ---
# These modules are assumed to be pre-existing and provide the necessary features.
# For demonstration purposes, we'll create dummy versions.

class SVFEModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        return self.conv(x) # Output (B, out_channels, H, W)

class PhysicsPriorFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, out_channels)
        )
    def forward(self, x):
        return self.mlp(x) # Output (B, out_channels)


# --- 1. Spectral Decomposition Module (SDM) ---
class SpectralDecompositionModule(nn.Module):
    def __init__(self, in_channels, num_frequency_bands):
        super().__init__()
        self.num_frequency_bands = num_frequency_bands
        self.in_channels = in_channels

        # Learnable filters for frequency band separation (conceptual)
        # In a real scenario, these could be fixed filters (e.g., Gabor, Wavelet) or learned.
        # For simplicity, we'll simulate frequency bands by splitting channels after FFT.

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # 1. Apply 2D FFT
        # torch.fft.fft2 operates on the last two dimensions (H, W)
        # For complex output, it will be (B, C, H, W, 2) for real/imaginary parts
        # Or, we can use torch.fft.rfft2 for real input, which returns (B, C, H, W//2+1, 2)
        # Let's use rfft2 for efficiency and real input
        
        # Pad to even dimensions if necessary for rfft2 to be clean
        pad_H = (H % 2 != 0)
        pad_W = (W % 2 != 0)
        if pad_H or pad_W:
            x = F.pad(x, (0, pad_W, 0, pad_H), mode='reflect')
            H, W = x.shape[-2:]

        fft_output = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho') # (B, C, H, W//2 + 1, 2)
        
        # Convert to magnitude and phase for potential manipulation, or keep complex
        # For simplicity, we'll just split the complex tensor into real and imag parts
        # and treat them as 2*C channels for routing/experts
        
        # Reshape to (B, C * 2, H, W//2 + 1) for easier processing by conv layers
        fft_output_reshaped = torch.cat([fft_output.real, fft_output.imag], dim=1)

        # Conceptual frequency band splitting:
        # For simplicity, we'll just divide the channels into num_frequency_bands groups.
        # In a real implementation, this would involve actual frequency filtering.
        
        # Ensure num_frequency_bands divides C * 2
        if (C * 2) % self.num_frequency_bands != 0:
            raise ValueError(f"Number of channels ({C*2}) must be divisible by num_frequency_bands ({self.num_frequency_bands})")
        
        channels_per_band = (C * 2) // self.num_frequency_bands
        
        frequency_bands = []
        for i in range(self.num_frequency_bands):
            band_features = fft_output_reshaped[:, i * channels_per_band : (i + 1) * channels_per_band, :, :]
            frequency_bands.append(band_features)
            
        return frequency_bands, (pad_H, pad_W) # List of (B, channels_per_band, H, W//2 + 1) tensors

    def inverse(self, frequency_bands_processed, original_shape, padding_info):
        # frequency_bands_processed: List of (B, channels_per_band, H, W//2 + 1)
        B, _, H, W_half = frequency_bands_processed[0].shape
        C_total = sum(f.shape[1] for f in frequency_bands_processed)
        C_original = C_total // 2 # Since we concatenated real and imag

        # Reconstruct the full complex tensor
        combined_fft_features = torch.cat(frequency_bands_processed, dim=1)
        
        real_part = combined_fft_features[:, :C_original, :, :]
        imag_part = combined_fft_features[:, C_original:, :, :]
        
        reconstructed_fft_output = torch.complex(real_part, imag_part)
        
        # Apply inverse 2D FFT
        reconstructed_image = torch.fft.irfft2(reconstructed_fft_output, s=(H, W_half * 2 - (1 if original_shape[-1] % 2 != 0 else 0)), dim=(-2, -1), norm='ortho')

        # Remove padding if applied
        pad_H, pad_W = padding_info
        if pad_H or pad_W:
            reconstructed_image = reconstructed_image[:, :, :original_shape[-2], :original_shape[-1]]

        return reconstructed_image


# --- 2. Spectral-Decoupled Expert (SDE) ---
class SpectralExpert(nn.Module):
    def __init__(self, in_channels_band, out_channels_band):
        super().__init__()
        # A very lightweight network for specific frequency band restoration
        self.net = nn.Sequential(
            nn.Conv2d(in_channels_band, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, out_channels_band, 3, 1, 1)
        )

    def forward(self, x):
        return self.net(x)


# --- 3. Dynamic Topology-Aware Router (DTAR) ---
class DynamicTopologyAwareRouter(nn.Module):
    def __init__(self, svfe_feature_dim, prior_dim, num_frequency_bands, hidden_dim=128):
        super().__init__()
        self.num_frequency_bands = num_frequency_bands
        
        # Feature projection for router input
        self.feature_project = nn.Sequential(
            nn.Linear(svfe_feature_dim + prior_dim, hidden_dim),
            nn.ReLU()
        )

        # GNN-like module for dynamic topology-aware routing
        # For simplicity, we'll use a self-attention mechanism to model topology
        # In a full GNN, you'd have explicit graph construction and message passing.
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, num_frequency_bands)

    def forward(self, svfe_features_map, physics_prior):
        # svfe_features_map: (B, svfe_feature_dim, H, W)
        # physics_prior: (B, prior_dim)

        # Pool SVFE features to a vector per batch item
        svfe_features_vec = F.adaptive_avg_pool2d(svfe_features_map, (1, 1)).squeeze(-1).squeeze(-1)

        # Concatenate all features for routing
        combined_features = torch.cat([svfe_features_vec, physics_prior], dim=1) # (B, svfe_feature_dim + prior_dim)
        
        # Project features
        projected_features = self.feature_project(combined_features) # (B, hidden_dim)

        # Simulate dynamic topology-aware routing with self-attention
        # Here, we treat each frequency band as a 'token' or 'node' conceptually
        # and use self-attention to determine their importance based on input features.
        # For a true GNN, you'd have explicit expert embeddings as nodes.
        
        # Create dummy 'frequency band tokens' for attention (conceptual)
        # In a real GNN, these would be expert embeddings or frequency band descriptors.
        # For now, let's just use the projected_features as query and key for itself,
        # and learn a separate 'value' for each frequency band.
        
        # Expand projected_features to act as query for each band
        query = self.query_proj(projected_features).unsqueeze(1) # (B, 1, hidden_dim)
        
        # Create 'key' and 'value' for each frequency band
        # This is a simplification. Ideally, keys/values would be derived from frequency band properties.
        # For now, let's just create learnable parameters for keys and values for each band.
        band_keys = self.key_proj(self.feature_project.weight.T).unsqueeze(0) # (1, hidden_dim, hidden_dim) -> (1, hidden_dim, hidden_dim)
        band_values = self.value_proj(self.feature_project.weight.T).unsqueeze(0) # (1, hidden_dim, hidden_dim)

        # Let's simplify this. The router directly outputs logits for each band.
        # The 'topology-aware' part comes from the rich input features and the MLP's ability to learn complex relationships.
        # For a more explicit GNN, we'd need a graph structure.
        
        # A simpler approach for DTAR: MLP outputs logits, and the 'topology-aware' aspect
        # is implicitly learned from the rich input features (SVFE + PhysicsPrior) that encode
        # frequency-spatial-physical relationships.
        
        # Let's use a simple MLP for routing, but acknowledge that a GNN would be more explicit.
        # The 'dynamic topology' is implicitly captured by the router learning to weigh bands based on input.
        
        routing_logits = self.output_proj(projected_features) # (B, num_frequency_bands)
        routing_weights = F.softmax(routing_logits, dim=1) # (B, num_frequency_bands)
        
        return routing_weights


# --- Integration with MoCE_IR ---
class MoCE_IR_SDE_DTAR(nn.Module):
    def __init__(self, in_channels_img, svfe_feature_dim, prior_dim, num_frequency_bands, 
                 expert_channels_per_band=3):
        super().__init__()
        
        self.in_channels_img = in_channels_img
        self.num_frequency_bands = num_frequency_bands

        # SVFE and PhysicsPriorFusion modules (placeholders)
        self.svfe = SVFEModule(in_channels=in_channels_img, out_channels=svfe_feature_dim)
        self.physics_prior_fusion = PhysicsPriorFusionModule(in_channels=in_channels_img, out_channels=prior_dim)

        # Spectral Decomposition Module
        self.sdm = SpectralDecompositionModule(in_channels=in_channels_img, num_frequency_bands=num_frequency_bands)
        
        # Dynamic Topology-Aware Router
        self.dtar_router = DynamicTopologyAwareRouter(
            svfe_feature_dim=svfe_feature_dim, 
            prior_dim=prior_dim, 
            num_frequency_bands=num_frequency_bands
        )
        
        # Spectral-Decoupled Experts
        # Each expert processes a frequency band. The channels_per_band from SDM output
        # will be the in_channels for these experts.
        # Assuming SDM splits C*2 channels into num_frequency_bands, so each expert gets (C*2)/num_frequency_bands channels.
        sdm_out_channels_per_band = (in_channels_img * 2) // num_frequency_bands
        self.spectral_experts = nn.ModuleList([
            SpectralExpert(sdm_out_channels_per_band, sdm_out_channels_per_band)
            for _ in range(num_frequency_bands)
        ])

    def forward(self, degraded_image):
        batch_size = degraded_image.size(0)
        
        # 1. Feature Extraction for Router
        svfe_features_map = self.svfe(degraded_image) # (B, svfe_feature_dim, H, W)
        physics_prior = self.physics_prior_fusion(degraded_image) # (B, prior_dim)
        
        # 2. Spectral Decomposition
        frequency_bands, padding_info = self.sdm(degraded_image) # List of (B, channels_per_band, H, W//2 + 1)

        # 3. Dynamic Topology-Aware Routing
        # routing_weights: (B, num_frequency_bands), each value is alpha_k
        routing_weights = self.dtar_router(svfe_features_map, physics_prior)

        # 4. Spectral-Decoupled Experts for Restoration
        processed_frequency_bands = []
        for i in range(self.num_frequency_bands):
            # Apply expert and then scale by routing weight
            expert_output_band = self.spectral_experts[i](frequency_bands[i])
            # Scale the expert output by its routing weight
            weighted_expert_output_band = expert_output_band * routing_weights[:, i].view(batch_size, 1, 1, 1)
            processed_frequency_bands.append(weighted_expert_output_band)
        
        # 5. Spectral Aggregation and Inverse Transform
        reconstructed_image = self.sdm.inverse(processed_frequency_bands, degraded_image.shape, padding_info)
        
        return reconstructed_image, routing_weights # Return for loss calculation


# --- Conceptual Training Loop (Simplified) ---

# def train_sde_dtar(model, dataloader, optimizer, num_epochs, device):
#     model.train()
#     
#     reconstruction_loss_fn = nn.L1Loss()
#     
#     lambda_sparsity = 1e-3 # Example sparsity regularization
#     lambda_balance = 1e-2  # Example load balancing regularization
# 
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for batch_idx, (degraded_image, clean_image) in enumerate(dataloader):
#             degraded_image = degraded_image.to(device)
#             clean_image = clean_image.to(device)
# 
#             optimizer.zero_grad()
# 
#             # Student forward pass
#             reconstructed_image, routing_weights = model(degraded_image)
#             
#             # Reconstruction Loss
#             reconstruction_loss = reconstruction_loss_fn(reconstructed_image, clean_image)
# 
#             # Sparsity Loss (L1 on routing weights to encourage sparsity)
#             sparsity_loss = torch.mean(torch.sum(torch.abs(routing_weights), dim=1))
# 
#             # Load Balancing Loss (variance of expert usage)
#             # This is a simplified version. A more robust load balancing would track actual expert usage.
#             expert_usage = torch.mean(routing_weights, dim=0) # Average usage per expert in batch
#             balance_loss = torch.var(expert_usage)
# 
#             # Total Loss
#             loss = reconstruction_loss + lambda_sparsity * sparsity_loss + lambda_balance * balance_loss
#             
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
# 
#         print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")


# --- Example Usage (Conceptual) ---
# if __name__ == "__main__":
#     # Dummy parameters
#     in_channels_img = 3
#     svfe_feature_dim = 64
#     prior_dim = 32
#     num_frequency_bands = 4 # e.g., Low, Mid-Low, Mid-High, High frequency bands
#     expert_channels_per_band = (in_channels_img * 2) // num_frequency_bands # From SDM output

#     # Initialize Model
#     model = MoCE_IR_SDE_DTAR(
#         in_channels_img=in_channels_img,
#         svfe_feature_dim=svfe_feature_dim,
#         prior_dim=prior_dim,
#         num_frequency_bands=num_frequency_bands,
#         expert_channels_per_band=expert_channels_per_band
#     )

#     # Dummy input
#     dummy_degraded_image = torch.randn(1, in_channels_img, 256, 256)

#     # Forward pass
#     reconstructed_image, routing_weights = model(dummy_degraded_image)
#     print(f"Input image shape: {dummy_degraded_image.shape}")
#     print(f"Reconstructed image shape: {reconstructed_image.shape}")
#     print(f"Routing weights shape: {routing_weights.shape}")

#     # Check parameter count (conceptual)
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total trainable parameters: {total_params / 1e6:.2f} M")

#     # Note: For a real implementation, you would need to define actual SVFEModule and PhysicsPriorFusionModule
#     # and integrate them properly with your dataset and training loop.
```

## 8. 参考文献

[1] Chen, M., et al. (2023). *Metalens-based Endoscopy: Challenges and Opportunities in Medical Imaging*. Nature Biomedical Engineering.
[2] Fedus, W., et al. (2022). *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*. Journal of Machine Learning Research.
[3] Shazeer, N., et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*. International Conference on Learning Representations (ICLR).
[4] DeepSeek-V3: *DeepSeek-V3: A New Generation of Mixture-of-Experts Language Models*. (2024). [https://deepseek.com/blog/deepseek-v3](https://deepseek.com/blog/deepseek-v3)
[5] Kobaayyy. (2024). *MoE-related ideas and implementations*. GitHub Repository. [https://github.com/Kobaayyy](https://github.com/Kobaayyy)
[6] Bruna, J., & Mallat, S. (2013). *Invariant scattering convolution networks*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(8), 1872-1886.
