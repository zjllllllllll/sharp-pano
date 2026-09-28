# 原版 SHARP：全景 RGB + 深度，六面浅注入

本目录仅保留原版 SHARP 原生外部深度输入路线。原始模型源码、权重和 RGB 未改动。

本页描述独立的历史六面批处理路线；当前网页第 3–5 步的定义见 [本地网页平台](../pano-web/README.md)。私有测试图、数值深度与生成结果不随源码发布。以下路径均相对仓库根目录。

## 输入与输出

- RGB：`pano-tests/{0.1,1.1,2.1,3.1,4.1}.png`。
- 深度：`depthmaster-results/{名称}.range_m.npy`，与 RGB 配准的数值径向距离，不是彩色深度图。
- 输出根目录：`sharp-sixface-results`。`shallow/` 为合并 PLY，`inputs/` 为切图与相机，`validation/` 为渲染，`reports/` 为报告。
- `work/{场景}/{面}.shallow.ply` 保留逐面世界坐标中间结果。

## 相机与深度约定

六面顺序为 front、right、back、left、up、down。每面 1536×1536，水平/垂直 FOV 均为 100°。采用 OpenCV 坐标：+X 右、+Y 下、+Z 前；六面相机位于同一全景原点，前向分别为 +Z、+X、−Z、−X、−Y、+Y。

焦距 `fx=fy=1536/(2*tan(50°))=644.428516744151`。像素边界坐标主点为 768，像素中心是 (列+0.5, 行+0.5)；同时保存像素索引坐标的内参，主点为 767.5。每面保存 K、c2w、w2c、FOV、尺寸和文件哈希。

RGB/深度沿同一世界射线采样。全景径向距离转换为透视相机 Z 深度：`camera_Z=range*unit_ray_z`。已知焦距通过 `disparity_factor=fx/image_width` 传入模型，不做焦距或位姿估计。

## 推理与融合

直接调用 `predictor(image, disparity_factor, depth=target_camera_z)`。原生 learned alignment、特征网络、预测头和 Gaussian composer 均照常执行，保留两层高斯，不额外改变预测深度。

高斯中心及协方差同步进行 NDC→相机→世界坐标变换。使用切图时确定的外参，无需 ICP。100° 面的 40°～50° 边缘进行余弦权重渐变，六面权重归一化，透明度按光学厚度融合：`alpha_new=1-(1-alpha)^weight`。

这不是多视图联合优化；各面的预测深度差异仍可能造成重影。不会按深度阈值静默删去异常远点。

## 环境与复现

权重：`models/sharp_2572gikvuh.pt`。

SHA-256：`94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d`。

原版源码版本：`1eaa046834b81852261262b41b0919f5c1efdd2e`。严格加载 1038 项权重，包括 156 项深度对齐权重。使用根目录安装脚本创建的 `.envs/runtime`；历史验证设备为 RTX 5070 Ti，FP32、PyTorch 2.8.0+cu128。

PowerShell：

```powershell
.\run_cuda_python.cmd sharp-sixface/run_pipeline.py
```

批处理前需先重新生成对应的 DepthMaster 数值深度。已完成的面经哈希验证后跳过。全新重跑使用新的 `--output-dir`；`--prepare-only` 仅准备切图，`--images 2.1` 仅处理指定场景。现有 PLY 不会被静默覆盖。

渲染使用项目内 gsplat 环境；在仓库根目录执行：

```powershell
.\run_cuda_python.cmd sharp-sixface/render_checks.py
```

已有渲染报告和 PLY 哈希相符时，复用单张结果并重建两列对照图。无报告时实际渲染 10 个视角。

独立验证：

```powershell
.\run_python.cmd sharp-sixface/audit_results.py
```

验证包括 8 项几何/协方差/PLY 单元测试、30 面相机与输入投影、5 个最终 PLY 的全部属性和精确逐面合并、50 张实际渲染及其文件来源。

## 已知限制

外部深度是估计值，且经过 learned alignment 和高斯前向后不会被严格保持。部分面偏差明显；完整 p90、p99、最大值见报告，不应只看中位数。原模型文档更推荐训练期提供深度、推理期采用自身单目深度，本路线是对外部深度接口的实际使用与评估。

中心深度不同于渲染期望深度。单张全景没有遮挡背面的观测；本流程没有补洞、稠密化或逐场景优化。原点/接缝视角的 RGB PSNR 是输入重现指标，不是新视角真实质量指标。
