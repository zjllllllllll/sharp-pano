# 本机 DepthMaster 全景推理

## 安装位置与版本

- 代码：`D:\sharp-pano\DepthMaster`
- 官方仓库：https://github.com/PolyU-VCLab/DepthMaster
- 上游代码提交：`c2ec897fa14803f05f211f6de49571801a4edfa1`
- 独立 Conda 环境：`depthmaster`（从本机 sharp 环境复制；未修改原环境）
- Python 3.13.15 / PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128
- 已验证 GPU：NVIDIA GeForce RTX 5070 Ti 16GB
- 新增推理依赖：opencv-python 4.11.0.86、einops 0.8.1、moderngl 5.12.0。
- utils3d 使用官方指定提交：`3fab839f0be9931dac7c8488eb0e1600c236e183`

权重：`checkpoints/depthmaster.pt`，1,541,004,203 字节。

官方来源：https://huggingface.co/VCLab-PolyU/DepthMaster

权重 revision：`add341b64ae9f90372b6483b334ac496a18fe081`

SHA-256：`e61e617acdd0b40bd2c217def5d60b0bd05352e6ae61d940a49c2374c6c75982`

`download_checkpoint.py` 支持分段续传并核对以上 SHA-256；运行前会跳过已校验的完整权重。

## 使用

```powershell
conda activate depthmaster
cd D:\sharp-pano\DepthMaster
python infer_panoramas.py --images 0.1.png 1.1.png 2.1.png 3.1.png 4.1.png
```

默认读取 `D:\sharp-pano\pano-tests`，输出到 `D:\sharp-pano\depthmaster-results`。
已有结果时默认拒绝覆盖。需要重新跑可指定另一个 `--output-dir`，或明确加 `--overwrite`。

无须激活环境也可直接使用：

```powershell
& C:\Users\admin\AppData\Local\miniforge3\envs\depthmaster\python.exe D:\sharp-pano\DepthMaster\infer_panoramas.py --input-dir D:\sharp-pano\pano-tests --output-dir D:\sharp-pano\depthmaster-results-new --checkpoint D:\sharp-pano\DepthMaster\checkpoints\depthmaster.pt --images 5.1.png
```

## 输出语义

每张图只输出以下三种文件，不产生网格、高斯、法线图或额外 RGB：

- `*.range_m.npy`：与原图同尺寸的 float32 径向距离，单位米；无效点为 NaN。
- `*.depth.png`：统一 0–10 米色标的深度预览，超过 10 米仅在预览中饱和；浮点深度未截断。
- `*.ply`：每个有效深度像素对应一个彩色点，二进制 little-endian XYZRGB。

PLY 坐标：相机中心为原点，X 向右、Y 向上、Z 向前。
点云严格按 ERP 像素中心射线乘径向距离反投影，没有平面拟合、尺度对齐或几何补全。
若以后注入透视版 SHARP，需要转换为该透视相机的 camera-Z，不能直接使用 range。

保持官方推理配置：六面联合预测，每面 518×518，FOV 95°，官方相机条件与软融合。
ERP 融合分辨率 2072×1036，然后上采样到输入 4096×2048；上采样不代表额外细节。
模型权重使用 strict=True 检查，无缺失或多余参数；保留原始模型预测，只屏蔽有效性概率 <= 0.5 的像素。

## Windows 兼容改动

1. `attention.py`：没有可用 flash-attn 扩展时调用 PyTorch SDPA；不改变权重或注意力公式。
2. `eval_panorama/eval.py`：将只用于数据集评估的导入延迟到 main，推理不依赖评估数据集。
3. 推理脚本用显式文件加载规避上游 `eval.py` 与 `eval/` 的同名导入冲突。

## 验证和预览

```powershell
python -m unittest -v test_infer_panoramas
python verify_results.py
```

验证全部点的 XYZ、RGB、深度数值、文件数量和尺寸；预览输出到代码目录 `preview/`，不污染结果目录。
点云预览仅为便于观察隐藏上部点并抽样；完整 PLY 未裁剪、未抽样。

## 相对深度可视化

已有米制深度无需重新推理，可以按每张图自身的 P1–P99 深度范围归一化显示：

```powershell
python export_relative_depth.py
```

输出到 `D:\sharp-pano\depthmaster-relative-depth`，保留原图尺寸：

- `*.relative.color.png`：相对深度彩色图，蓝近、红远。
- `*.relative.gray.png`：相对深度灰度图，白近、黑远。

两版使用同一个线性归一化范围，范围外各约 1% 的像素仅在显示时饱和；无效像素为黑色。
归一化参数嵌入 PNG 元数据，每张图独立定标，不能用颜色跨图比较米制距离。
这只是显示对比度的调整，不是重新估深或细节增强。原始 `.range_m.npy`、`.ply` 和绝对深度预览都不改变。
反投影和注入 SHARP 仍应使用原始米制深度，不应使用这些预览 PNG。
三列对照图位于 `preview/relative-depth-comparison.png`。
