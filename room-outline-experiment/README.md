# 点云轮廓与采样相机实验

> 本文保留历史实验记录。当前应用安装和启动见 [根目录 README](../README.md)；当前统一使用根目录的 `run_python.cmd` / `run_cuda_python.cmd`。场景图片和历史 runs 不随源码发布。

独立离线实验与只读网页栏目，不改变原来的五步推理、Mask 或高斯加密逻辑。

## 查看结果

- 当前网页：`http://127.0.0.1:8769/camera-lab`。旧端口可能仍展示历史版本；原五步服务不变。
- 当前数据：`runs/20260922-adaptive-sampling/report.md`，47 个场景。历史 unknown-v1 / unknown-v2 / 旧点云预览保留，不覆盖。
- `bev.png`：俯视点云；`overlay.png`：叠加轮廓；`cameras.png`：内缩区域、采样点和编号。
- `panoramas/C001.png`：编号相机应用 Mask 后的低清全景；同名 `-mask.png` 是修补 Mask，`-coverage.png` 是最终保留区域。
- `sample-cameras.json`：最终 cameras、完整几何 all_cameras、固定规则与排除原因。`candidate-cameras.json` 只含最终候选；原 `cameras.json` 仍是 SHARP 六面相机，不覆盖。
- `sampling-validation.json`：当前固定筛选及全量图像/深度产物核验；`browser-validation.json` 为浏览器核验。它们不代表墙线真值或相机无碰撞认证。

## 处理链路

使用现有 `pano-web/worker.py` 的 `depth_stage` 和 `cloud_stage`，输出 DepthMaster 径向深度、原生浅注入后的 SHARP 双层初始化中心。为批处理缓存模型加载，但不改变推理函数、种子和参数。不做最终高斯前向，也不做既有法向/KNN Mask 或点云加密。

轮廓只依赖点云俯视几何及连续高度支撑：5cm 栅格、24 个高度层、多方向径向证据搜索、闭合边界简化和受限直线拟合。绿色实线是较强几何支撑，橙色虚线是较弱/推断段。当前是相对原相机的星形可见包络，不能保证恢复遮挡后的完整房间，也不保证所有边界都是墙。`bev-full.png` 保留完整投影范围，便于检查离群点。

采样固定使用原相机为原点的 0.5m X/Z 格网，Y=0。内缩从 0.5m 每次减 0.05m，直到原点八邻域有至少 4 点，或 offset <0.1m（最后 0.05m）。仅剩原点报错。再固定保留 Mask ≤15% 的点，最后取原点所在四邻接连通分量；只可沿前后左右移动 0.5m，不跨对角或被删点。筛选后仅剩原点同样报错，不再放宽参数。移除调试滑块，旧浏览器设置不生效。详见 [ADAPTIVE-SAMPLING.md](ADAPTIVE-SAMPLING.md)。长度依赖预测深度尺度，规则不保证真实室内或无家具碰撞。

当前渲染入口已切换为 **1024×512 原始 RGB-D 可靠表面 + 未知空间遮挡 Mask**，不是最终高斯渲染。默认展示应用 Mask 的图，提供遮挡前表面图、最终 Mask、叠加和原因切换；没有 AI 修补。展开/自由转动两种模式显示同一版本，后者只改变朝向，不移动相机。实现、限制和独立批量命令见 [UNKNOWN-MASK.md](UNKNOWN-MASK.md)。旧点云渲染输出保留用于对照，不作为新方案缓存。

## 复现与续跑

在本目录执行（新数据集先为 `prepare` 指定新的 `--run` 目录）：

```powershell
..\run_python.cmd batch.py prepare --run runs/new-run
..\run_python.cmd batch.py depth --run runs/new-run
..\run_python.cmd batch.py cloud --run runs/new-run
..\run_python.cmd postprocess.py --run runs/new-run
..\run_python.cmd finalize_candidates.py --run runs/new-run
```

已有输出可续跑。不要在硬链接目录内重跑模型数组。当前 47 场景使用 `finalize_candidates.py` 更新筛选和图表、`validate_adaptive.py` 做全量核验，不重新生成全景。网页默认读取 `runs/20260922-adaptive-sampling`；已运行的历史端口并不一定指向该目录。

## 历史：0.5m 修订版（不是当前候选规则）

`resample.py` 在独立目录生成新版，按完全相同的三维相机位置复用旧版渲染，不能用编号匹配。它不重跑深度、SHARP 或轮廓拟合。新旧相机编号可能不同。

发布后 `runs/20260921-all` 为新版，`runs/20260921-inset-1p5` 保留旧版。旧版全部相机图、全景和元数据保留。新旧版的大型深度/点云/初始化数组通过 NTFS 硬链接只读复用；**不要在这些版本目录内重跑或覆盖推理数组**，需要重跑模型时用全新的 run 目录。

`verify_resample.py` 检查新规则、全部全景、旧版文件哈希以及复用图像和三维位置的一致性。前端标签读取结果中的参数，图片 URL 带版本标记，以免相同编号读到旧缓存。

Shapely 2.1.2 单独安装在 `.deps`，没有修改现有推理环境的依赖。

电脑重启后，在项目根目录单独启动实验网页：

```powershell
.\run_python.cmd room-outline-experiment/serve_camera_lab.py --port 8769
```
