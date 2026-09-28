# 单全景快速结构引导 3DGS 完整房间重建研究方案

## 1. 目标与问题定义
从**单个房间的一张 360° 全景图（或极少量图像）**出发，无需用户额外补拍，快速生成结构完整、主要遮挡区域得到合理补全、无明显空洞与接缝、可从新视角稳定观察的 3DGS。

本研究不要求不可见区域严格恢复 Ground Truth，而采用：**已观测区域尽量保持真实；不可观测区域生成几何合理、语义合理、视觉可信且与已有场景连续的内容。**

核心问题：**能否利用房间结构先验、主动视角选择、深度补全和快速单目 Gaussian lifting，以最少生成视图和尽量少的全局优化，实现快速、结构一致的单全景 3DGS 补全？**

暂定名称：**Fast Structure-Guided 3DGS Completion from a Single Panorama**。

单张全景虽然覆盖 360°，但只有一个相机中心。床、柜体、沙发等会遮挡其后的墙面、地面和其他结构，因此初始重建只能恢复可见表面。隐藏区域从未被观测时，其真实内容不能由输入唯一确定，所以目标应是：`Preserve observed reality + plausibly complete unobserved space`。

## 2. 已有三个技术模块

### 2.1 点云 + 轨迹 + 图像语义的户型结构推理
几何分支：`Point Cloud → Plane / Normal Estimation → Wall / Floor / Ceiling Candidates`。

语义分支：`RGB → Semantic Segmentation → Camera Pose + Trajectory → Project Semantics to Point Cloud`。

两者融合得到墙、地、顶及家具位置等结构化房间模型。它为新系统提供 **Room Structural Prior**：不仅知道“哪里没有 Gaussian”，还能判断“哪里理论上应该存在墙、地、顶”。

### 2.2 已有 3DGS 局部补洞
```text
Incomplete 3DGS
  ↓ Render RGB + Depth + Hole Mask
RGB Hole → FLUX/GPT → Completed RGB
Depth Hole → LingBot-Depth → Completed Depth
  ↓
Depth-constrained SHARP
  ↓
Generated Gaussians
  ↓
Extract Hole Gaussians → Merge into Original 3DGS
```
职责：2D 生成模型负责 Appearance；深度补全负责 Missing Geometry；原始 GS 深度约束已知几何；SHARP 快速 lifting；最终只提取洞内 Gaussian，保护正确区域。

### 2.3 3DGS 全景修复
方法 A：`ERP → 6 Perspective Faces (FOV>90°) → 独立修复/超分 → Overlap Feather → ERP`。

方法 B：在相邻 cubemap face overlap 中用 Optical Flow 对齐后融合。

这套经验可迁移到 RGB、Depth 以及生成视图的一致性处理。

## 3. 总体 Pipeline
```text
Single 360° Panorama
 → Cubemap / Perspective Decomposition
 → RGB Restoration (optional)
 → Depth Estimation
 → Cross-face Depth Alignment + Fusion
 → Consistent RGB-D Panorama
 → Fast Gaussian Lifting
 → Initial Partial 3DGS G₀
 → Room Structure Inference
 → Room Envelope + Semantic Structure
 → Missing Surface Detection
 → Next-Best-View / Minimal View Planning
 → Render Partial RGB + Depth + Hole Mask
 → Structure-guided RGB/Depth Completion
 → Depth-constrained SHARP
 → Local Gaussian Extraction + Merge
 → Coverage / Quality Evaluation
 → 必要时迭代 1–3 次
 → Complete / Plausible Room 3DGS
```
核心原则：**不是生成尽可能多的新视图，而是让每一个生成视图最大化新增三维信息。**

## 4. Stage 1–3：从全景建立初始 Partial 3DGS

### 4.1 全景拆分
输入 ERP 全景 `P₀`，转换为六个 perspective faces：`P₀ → {I₁…I₆}`。建议 FOV≈100°，制造稳定 overlap，用于 RGB 融合、深度跨面约束、光流/特征对齐和 seam 抑制。

### 4.2 深度估计与跨面统一
对每个 face 使用 Depth Anything 等模型：`Iᵢ → Dᵢ`。独立预测可能存在 scale、shift 和边界不一致，因此给每面求 `Dᵢ'=sᵢDᵢ+tᵢ`，利用相邻 overlap 联合优化，使 `Dᵢ'≈Dⱼ'`。可加入 robust loss 降低物体边界、透明/反射区域的影响。

对于墙、地、顶区域进一步施加平面约束 `ax+by+cz+d=0`。最终深度来自：**Monocular Depth + Cross-face Consistency + Room Plane Prior**。

### 4.3 初始 Gaussian lifting
若 SHARP 以 perspective image 为输入，可对六面分别 lifting：`Iᵢ + Dᵢ → Gᵢ`，利用 cubemap 已知外参 `Tᵢ` 统一：`G₀ = Union(TᵢGᵢ)`。

随后进行 duplicate pruning、overlap depth filtering、opacity/scale 异常清理、redundant merge、boundary artifact removal。`G₀` 的目标不是完整，而是高质量表达**真实可见部分**。

## 5. Stage 4：Room Structural Prior 与真正的“洞”
从 `G₀`、深度或采样点云接入已有户型推理模块，得到 Wall / Floor / Ceiling / Furniture 和 Room Envelope。

定义理论结构表面 `S_room`，当前可靠观测表面 `S_obs`：`S_missing = S_room - S_obs`。

关键区别：**Empty 3D Space ≠ Missing Surface**。房间中央没有 Gaussian 是正常的；柜子后面理论上应该存在墙而没有 Gaussian，才是结构洞。

## 6. Stage 5：Missing Region 分类

### 6.1 Structural Hole
已确定属于墙、地、顶等参数化结构的洞，其几何不需要 AI 猜。若墙面为 `Π: ax+by+cz+d=0`，从虚拟相机发射 `r(λ)=o+λd`，通过 ray-plane intersection 直接得到洞内深度。

因此：`Structural Hole → Analytical / Geometric Depth + Generated Appearance`。

### 6.2 Non-structural Hole
家具后方复杂未知结构、非规则表面、语义不确定区域，再使用 `RGB Inpainting + LingBot Depth Completion`。

原则：**能由确定几何求出的信息，不交给生成模型猜。**

## 7. Stage 6：Next-Best-View（NBV）
在 Room Envelope 内采样合法虚拟相机 `C={c₁…cₙ}`，要求不穿墙、不与家具碰撞、高度合理、距离表面合理、避免极端视角。

评分可定义为：

`Score = α·MissingArea + β·ViewQuality - γ·Collision - δ·Distortion - ε·Redundancy`

选择 `c* = argmax Score(c)`，即当前能暴露最多有效缺失表面的虚拟观察位置。

进一步可做 **Minimal-View Set Cover**：寻找最少相机集合，使 Missing Surface Coverage 达到如 95%，将多轮串行生成转为少量并行生成。这是方案与重型 progressive reconstruction 拉开速度差距的重要部分。

## 8. Stage 7–8：虚拟视角补全并回写 3D
从 `c*` 渲染：`R(c*,Gₖ) → {I_partial,D_partial,M_hole}`。

严格遵循：洞外 RGB 不改；洞外 depth 不改；mask feather 向洞内进行；已知 Gaussian 默认 immutable。

RGB：`I_partial + M_hole + Structure/Semantic Condition → FLUX/GPT → I_complete`。

Depth：Observed Region 使用原始 GS Depth；Structural Hole 使用解析 Room Geometry Depth；其他 Hole 使用 LingBot-Depth。

核心思想：**生成模型主要负责“这个表面看起来是什么”，结构模型负责“这个表面在三维空间里在哪里”。**

随后：`I_complete + D_complete → Depth-constrained SHARP → G_candidate`。要求新 Gaussian 在当前相机下投影深度与 `D_complete` 一致，避免单目 lifting hallucinate 与现有尺度不一致的结构。

只提取洞内：`G_patch = G_candidate ∩ M_hole`，然后 `Gₖ₊₁ = Gₖ ∪ G_patch`。边界处理包括 spatial clipping、depth consistency filtering、duplicate pruning、opacity transition、overlap merge、floater removal。

原则：**Known region is immutable by default; only missing region is editable.**

## 9. Stage 9：迭代与停止
每轮重新计算 Missing Surface 与 novel-view hole。终止条件可综合：Structural Coverage > 95%；novel-view hole area 小于阈值；新一轮 Information Gain 很小；达到最大迭代次数（如 2–4）；剩余区域价值低或无法可靠补全。

目标不是强求 100%，而是在**完整度、视觉质量、运行时间**之间找到最优停止点。

## 10. 与 Pano2Room 的核心区别
Pano2Room 已证明“单全景 → 完整房间”方向可行。本方案不重复这一贡献，而强调：

1. **Structure-Guided**：利用 room layout、plane、semantic prior，很多 missing geometry 可解析求解；
2. **Minimal Active Views**：不大量生成 novel views，而是求信息增益最大的最少视图；
3. **Feed-forward Local Gaussian Completion**：利用 SHARP 等快速 lifting，只生成局部 patch，避免每轮重新优化整个 scene；
4. **Preserve Known Geometry**：已知区域默认不可修改；
5. **Fast Completion**：研究目标从 “Can it reconstruct?” 转为 “Can it reconstruct sufficiently well with substantially lower latency and computation?”。

## 11. 核心算法抽象
`Gₖ₊₁ = Gₖ ∪ Completion(Render(Gₖ,Cₖ),S_room)`

其中：`Cₖ = argmax_C InformationGain(C | Gₖ,S_missing)`。

循环为：建立当前世界模型 → 判断缺失 → 主动选择最有价值的虚拟视角 → 只生成未知区域 → 提升回 3D → 合并 → 重新评估。

## 12. 评价指标

### 已知区域保真度
- RGB PSNR / SSIM / LPIPS；
- Depth Error；
- Gaussian displacement；
- geometry consistency。

### 完整度
- Room Structural Coverage；
- Missing Surface Area；
- novel-view hole ratio。

### 多视图一致性
- overlap RGB/depth error；
- ghosting / floater 数量；
- novel-view consistency。

### 几何合理性
- Wall/Floor/Ceiling plane error；
- room envelope violation；
- object-wall penetration；
- impossible geometry ratio。

### 效率
- Total Runtime；
- Number of Generated Views；
- Number of SHARP Calls；
- GPU Memory；
- Iteration Count；
- Added Valid Surface Area / Runtime。

## 13. Ablation Study
建议测试：
1. 无 Room Prior vs 有 Room Prior；
2. Random View vs NBV；
3. Greedy NBV vs Minimal-View Set Cover；
4. 全部 LingBot Depth vs Structural Hole 解析几何；
5. 无 Depth Constraint SHARP vs Depth-constrained SHARP；
6. 全局重建 vs Local Gaussian Merge；
7. Feather overlap vs Optical Flow alignment；
8. 单轮 vs 多轮 completion；
9. Cubemap Depth Anything vs 原生 panorama depth。

## 14. 最小可行实验（MVP）

### MVP-1：单全景 → 初始 GS
选 5–10 个单房间；全景拆六面；深度估计与融合；SHARP lifting；检查尺度、接缝和六面 merge。

### MVP-2：Room Structural Hole Detection
只做 `G₀ → Room Envelope → Missing Wall/Floor/Ceiling`，验证 Empty Space 与 Missing Surface 的区分。

### MVP-3：NBV
暂不生成内容，只验证选出的虚拟相机是否最大化暴露 Missing Surface。

### MVP-4：接入已有补洞模块
对 NBV 渲染图运行 RGB completion + Depth completion + SHARP + Local Merge，统计一轮增加多少有效 coverage。

### MVP-5：完整闭环
最多运行 2–3 轮，比较完整度、视觉质量、总耗时和速度/质量 trade-off。

## 15. 主要风险与应对

### 初始单目深度错误传播
错误的 `G₀` 会影响 Room Envelope、NBV 和后续补全。应加入跨面 depth consistency、room plane prior 和 uncertainty。

### 生成错误被迭代放大
第一轮 hallucination 可能在下一轮被当成真实世界。建议每个 Gaussian 记录 provenance / confidence：Observed、Structure-inferred、AI-generated，后续给予不同权重。

### 家具隐藏几何复杂
第一阶段明确限定：重点保证 Room Envelope 与大尺度视觉完整性，不要求隐藏家具结构严格正确。

### 多虚拟视角生成不一致
通过最小视图数、overlap conditioning、已有 Gaussian 作为下一视图约束、局部 merge 和 confidence weighting 抑制。

### 系统再次变重
必须把 runtime 当作一等指标。每增加一个模型/步骤，都验证其质量收益是否大于时间成本。

## 16. 推荐研究路线

**Phase A — 工程闭环：**先用已有模块完成可运行版本，不追求端到端创新。

**Phase B — 速度优化：**重点研究 NBV、Minimal-View、Structural Depth、Local Merge，减少生成次数和全局优化。

**Phase C — 一致性优化：**研究跨视角 RGB/Depth consistency、Gaussian provenance 和 uncertainty。

**Phase D — 3D-native 长期方向：**当前方案仍存在 `3D → 2D completion → 3D`。长期探索：

```text
Incomplete 3DGS
      ↓
3D Encoder / Gaussian Tokenization
      ↓
3D Latent
      ↓
3D Completion / Generative Prior
      ↓
Decoder
      ↓
Complete 3DGS
```

因此，本研究既可以形成短期可落地的生产方案，也能自然延伸到 **Gaussian-native / 3D latent completion**。

## 17. 一句话总结
> **从单张全景建立一个可信但不完整的三维世界模型；利用房间结构判断真正缺失的表面；主动寻找最有价值的虚拟观察位置；只对未知区域生成 RGB 与几何；快速提升回 Gaussian 并局部合并；以最少的生成视图迭代得到结构完整、视觉可信且运行足够快的单房间 3DGS。**
