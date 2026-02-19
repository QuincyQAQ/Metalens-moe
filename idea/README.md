# 创新点方案记录目录

本目录记录了所有讨论过的创新点方案，包括已实现和备选方案。

## 目录结构

```
idea/
├── README.md                          # 本文件
├── all_innovations_summary.md         # 所有创新点方案总结
├── innovation1_sv_frequency_embedding.py  # 创新点1代码实现
├── innovation2_physics_prior_fusion.py    # 创新点2代码实现
├── innovation3_ccr_dwr.py                 # 创新点3候选方案：CCR-DWR
├── innovation3_cgm_cr_ei.md               # 创新点3方案：CGM-CR-EI（已实现）
├── innovation3_hr_sea_dlb.md               # 创新点3方案：HR-SEA-DLB（已实现）
├── innovation3_ced_cr.md                  # 创新点3方案：CED-CR（待实现）
└── ... (其他备选方案文档)
```

## 已实现的创新点

### 创新点1: 空间变异频率诊断 (SVFD)
- **状态**: ✅ 已实现并集成
- **文件**: `MoCE_IR_S_SV_PhysicsPriorFusion.py`
- **核心模块**: `SpatiallyVariantFreqEmbedding`

### 创新点2: 多模态物理先验融合专家 (PGHC)
- **状态**: ✅ 已实现并集成
- **文件**: `MoCE_IR_S_SV_PhysicsPriorFusion.py`
- **核心模块**: `HeteroExpert`, `PhysicsPriorFusion`

### 创新点3候选方案

#### 方案A: 因果生成模型驱动的逆事实路由与专家干预 (CGM-CR-EI)
- **状态**: ✅ 已实现
- **文件**: `net/MoCE_IR_S_SV_PhysicsPriorFusion_CGM_CR_EI.py`
- **训练脚本**: `train_multiple_models.sh` 中已包含
- **详细文档**: `innovation3_cgm_cr_ei.md`
- **特点**: 可解释性强，理论深度高，泛化能力好

#### 方案B: 基于哈希路由的稀疏专家激活与动态负载均衡 (HR-SEA-DLB)
- **状态**: ✅ 已实现
- **文件**: `net/MoCE_IR_S_SV_PhysicsPriorFusion_HR_SEA_DLB.py`
- **训练脚本**: `train_multiple_models.sh` 中已包含
- **详细文档**: `innovation3_hr_sea_dlb.md`
- **特点**: 显存友好，计算效率极高，适合大规模部署

#### 方案C: 协同竞争路由与动态专家权重重缩放 (CCR-DWR)
- **状态**: ✅ 已实现
- **文件**: `net/MoCE_IR_S_SV_PhysicsPriorFusion_CCRDWR.py`
- **训练脚本**: `train_multiple_models.sh` 中已包含
- **详细文档**: `innovation3_ccr_dwr.py`
- **特点**: 平衡性能和效率

#### 方案D: 基于对比学习的专家解耦与协同路由 (CED-CR)
- **状态**: ⏳ 方案已设计，代码待实现
- **详细文档**: `innovation3_ced_cr.md`
- **特点**: 解决专家同质化问题，实现专家协同

## 快速导航

- **查看所有方案总结**: [all_innovations_summary.md](all_innovations_summary.md)
- **查看CGM-CR-EI详细方案**: [innovation3_cgm_cr_ei.md](innovation3_cgm_cr_ei.md)
- **查看HR-SEA-DLB详细方案**: [innovation3_hr_sea_dlb.md](innovation3_hr_sea_dlb.md)
- **查看CED-CR详细方案**: [innovation3_ced_cr.md](innovation3_ced_cr.md)

## 使用建议

1. **追求可解释性和理论深度**: 选择 CGM-CR-EI
2. **追求极致效率和显存友好**: 选择 HR-SEA-DLB
3. **平衡性能和效率**: 选择 CCR-DWR
4. **解决专家同质化问题**: 选择 CED-CR（待实现）

## 更新记录

- 2024-XX-XX: 创建文档目录，记录所有创新点方案
- 2024-XX-XX: 添加CGM-CR-EI、HR-SEA-DLB、CED-CR方案详细文档
