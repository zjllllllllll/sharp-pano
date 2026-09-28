# 低清全景：未知空间遮挡 Mask 实现

实现依据：[原设计记录](../novel-view-occlusion-mask-plan.md)。不改深度、SHARP、点云清理、加密、轮廓或已采样相机。

## 实际计算

1. 读取原图和 `depth.npy` 径向距离，不使用被填补的 `conditioning.npy`，不把 SHARP 第二层或加密点当作新增观测。
2. 每个有效深度像素保留完整角覆盖。连续区域共享角点；深度突变处分别保留两侧独立的小表面，不跨前后景连接，也不挖掉轮廓。无效像素不提供可靠表面。
3. 原相机到径向深度之前组成星形空域。单独建立空域边界的 BVH，用第一次相交表示新射线第一次进入未知空间。它只是加速查询的边界，不是真实物体模型，不会输出为高斯。
4. 对新相机每条射线分别得到最近可靠表面和首次未知边界的距离。未知在先则 Mask；没有可靠表面也 Mask；有效表面在先则保留。不会因为墙后存在未知空间就把墙一起删掉。
5. 无效/被过滤的方向用不确定区域处理；相机不在已确认空域时整图保守 Mask，并记录原因。
6. 每个输出像素 2×2 子射线；任意一条需修补，整个像素进入 Mask。输出依旧是 1024×512。

没有固定步长的体积射线采样，因此不会因步长直接跨过代理网格中的细窄遮挡。但是代理网格仍是离散近似，不承诺连续锥束的严格无漏检。

## 分辨率与容差（v2-footprints，非已证明最优）

- 几何代理宽度上限 2048。本批 4096×2048 深度按 2×2 块降到 2048×1024。
- 连续块取均值；有前后景跳变的块不混合，未知边界保留块内最小距离，避免细前景被均值抹掉。
- 不连续阈值：`max(0.05m, 3% × 局部近深度)`。
- 取消 v1 的深度边缘扩张及三角面整圈删除。深度断层的径向连接壁仅闭合未知体积，绝不作为已观测表面；不会把近处深度横向扩展到相邻有效像素。
- 表面径向容差：`0.02m + 0.5% × 深度`；相交距离比较容差 0.001m。
- ERP 左右周期连接，南北极有实际封口。对近乎共面的径向壁相交，用 float64 复核射线命中位置与三角形重心坐标位置是否一致；仅对漏交/不一致的射线做约 1e-7～3e-5 弧度级微扰重试（最多约 0.01 个代理像素）。仍失败则标记不确定，绝不当作已知空气。所有视点使用同一规则。
- 本批没有已有清理 Mask sidecar，可信度来自原始深度的有效性与连续性。未来若提供 `depth-valid.npy`，必须与原始 ERP 对齐，False 区域会被标为不确定，不会当作空气。

新视点仍会保守多补；但有效原视点不允许因边缘处理新增缺口。单目深度本身的错误仍会影响结果。

## 2026-09-22 修复与编号

- 修复前 `0.1` 原相机 C026 的 21,906 / 524,288 像素被 Mask（4.1782%）；其中 21,887 个包含“无可靠表面”，主要来自边缘表面被删除。
- v2 不特判原视点、不直接贴回原图。原视点和新视点使用同一套表面与未知体积相交流程。
- 每个完整有效的原视点必须做到 Mask 为零，否则批量渲染直接报错；验收不再排除“无可靠表面”的像素。
- 无效近端盖及细长径向壁用更严格的命中位置复核；仅在确定命中不成立时，以 10 微米余量跨过该伪命中继续沿原射线查询，失败仍标不确定。原视点最终 Mask 必须逐像素等于原深度有效性缩减后的结果。7.4 的 969 个无效源深度只对应 84 个输出 Mask 像素，不再扩大到 1,277 个。
- 本批 46 个全有效深度场景保留实际产出它们的代码哈希与快照；无效深度场景 7.4 使用后续修复重出。新增路径在全有效深度时不执行，12 个相机测试验证所有图像、Mask 和深度逐位相等；校验同时确认这 46 张源深度确实全有效。没有把旧产出伪标为新代码重新计算。
- 原相机统一 C000；半径为 `max(abs(ix),abs(iz))` 的方形圈从内向外编号。每圈从正上方 (+Z) 顺时针，缺失格点跳过、编号连续。
- 旧采样坐标不移动。没有原视点的场景添加一个 `reference_only` C000，不把它声称为满足内缩约束的新采样点。
- `previous_id` 保留旧编号对应关系；俯视图、下拉框、按钮和渲染文件统一新编号。
- 新版保存在 `runs/20260922-unknown-v2`，v1 结果和原始数据保持不变。

## 文件

每个相机 `panoramas/Cxxx`：

- `.png`：应用 Mask 后的预览，暗色为待修补区域。
- `-raw.png`：遮挡前的可靠表面 RGB 重投影，非旧点云渲染，也非最终高斯。
- `-mask.png`：白色为最终待修补 Mask。
- `-overlay.png`：原因颜色叠加。
- `-unknown.png`：未知体积先于可靠表面命中。
- `-holes.png`：无可靠表面。
- `-uncertain.png`：不确定路径或相机空域未确认。
- `-reasons.png`：原因彩色图；蓝→紫→红→黄为显示优先级。
- `-reason-bits.png`：uint8 原因位：1 无表面、2 未知先遮挡、4 路径不确定、8 相机未确认。
- `-coverage.png`：白色为最终保留区域，是最终 Mask 的反相，不是高斯 alpha。
- `-depths.npz`：可靠表面与未知边界的径向距离；2×2 子射线分别取最小值用于诊断。两张最小值不一定来自同一子射线，不能仅用这两个缩减深度反推最终 Mask。

`sample-cameras.json` 记录每个相机空域状态、各原因占比、参数、输入哈希、代码哈希和耗时。原因可重叠，占比不能直接相加。

## 批量与验证

```powershell
& 'C:\Users\admin\AppData\Local\miniforge3\envs\depthmaster\python.exe' -m pip install --target .render-deps --no-deps -r render-requirements.txt
& 'C:\Users\admin\AppData\Local\miniforge3\envs\depthmaster\python.exe' rerender_unknown.py --workers 1
& 'C:\Users\admin\AppData\Local\miniforge3\envs\depthmaster\python.exe' -m unittest test_occlusion test_experiment
& 'C:\Users\admin\AppData\Local\miniforge3\envs\depthmaster\python.exe' validate_unknown.py
```

独立目录生成，完整输出和指纹一致时可续跑。只保留模型大数组的只读 NTFS 硬链接，不覆盖源目录。默认 `rerender_unknown.py` 用 `20260921-unknown-v1` 作为输入，在 `20260922-unknown-v2` 生成结果。

只读预览任一版本（不会更改原五步任务的运行状态）：

```powershell
& 'C:\Users\admin\AppData\Local\miniforge3\envs\sharp\python.exe' serve_camera_lab.py --port 8768 --run D:\sharp-pano\room-outline-experiment\runs\20260922-unknown-v2
```

该命令查看历史 unknown-v2 渲染版本；当前固定候选采样使用 `serve_camera_lab.py --port 8769`（默认 adaptive-sampling 数据集），访问 `http://127.0.0.1:8769/camera-lab`，规则见 ADAPTIVE-SAMPLING.md。

实现使用 [Embree Python 封装](https://github.com/trimesh/embreex) 的最近三角形相交查询；空域建模与判定是本项目的实现，不是该库提供的场景补全算法。
