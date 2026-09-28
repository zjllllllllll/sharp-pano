# 固定候选相机规则

相机仍与原相机同高、0.5 m 点阵、原相机 C000、正方形圈从上方顺时针编号。本轮不实现逐视角修补，也不改深度、SHARP 或未知空间 Mask 算法。

## offset 规则

从 0.5 m 起，每次减 0.05 m。只统计原点八邻域 `max(abs(ix), abs(iz)) == 1`，不把原点本身或远处候选算进去。该八邻域有至少 4 个点就停止；否则继续直到 offset 小于 0.1 m（默认序列最后是 0.05 m）。每个场景保存全部尝试和实际 offset。除 C000 参考位外，所有点都严格满足最终内缩边界。

达到下限仍不足 4 个邻点但有其他点时，保留并显示警告；若只有原相机，抛出 `SamplingError`，网页显示明确采样失败，不能当作成功采样。

## 固定 Mask 与连通筛选

按渲染后最终二值 ERP Mask 的白像素数 / 总像素数比较（不是角度加权）。固定严格大于 15% 排除，等于保留；C000 始终作为参考保留。随后，在通过阈值的格点中，以 C000 为起点只允许前后左右移动一个格子（0.5 m），取能到达的全部点。对角相邻不算直接连通，不可跨越缺失点、超限点或待计算点。该规则没有滑块，也不读取旧的 localStorage 阈值。

绿点为最终候选，红叉为 Mask 超限，橙叉为与原点不连通，蓝点为原相机，灰点为待计算。相机菜单、上下切换、后端数据和导出 JSON 使用同一份最终集合。叉号可点击检查历史预览，但该点不加入候选或导出。俯视图导出为嵌入底图的 SVG；离线 cameras.png 使用相同分类。保留原编号，不重编号、不改位置、不删除历史预览。

顺序严格为：几何自适应 offset → Mask ≤15% → 原点四邻接分量。八邻域计数只用于第一阶段；最终邻点不足 4 个也不会再次放宽 offset。几何阶段只有原点抛出 SamplingError 并保存 sampling_error；最终筛选只剩原点保存 selection_error，网页明确报错。未完成渲染时不把“等待 Mask”误报为最终失败。

Mask 大并不严格等于房间外：室内遮挡严重处也可能被排除，门外清晰且连通的点仍可能留下。该规则不是语义、墙面识别或无碰撞认证。

## 数据与复现

新结果：`runs/20260922-adaptive-sampling`，旧结果 `runs/20260922-unknown-v2` 保留。已有相机按精确位置复用全部不可变图像与深度；仅新增位置运行原有 renderer。每个相机记录真实生成代码指纹，旧全有效深度版本的等价性证明保留在 implementation 目录。修改当前 renderer 不会伪造旧图的来源。

`sample-cameras.json` 的 cameras / count 是最终候选，all_cameras / geometric_camera_count 是完整几何点池；每个点有 selection_state。selection 记录固定参数、排除原因与 ID。neighbor_count 始终是 offset 阶段的八邻域计数，不是最终通过筛选的数量。candidate-cameras.json 及网页导出仅含最终候选。原 cameras.json 仍为 SHARP 六面相机，不能覆盖。

已有全量预览使用 depthmaster 环境运行 `finalize_candidates.py`，备份旧元数据与图表到 selection-history/before-mask15-connected，再更新候选、俯视图、总览和报告，不重跑渲染。新数据由 `cameras.process` 完成渲染后执行同一筛选；`adaptive_sampling.py --workers 1` 负责复用/补渲染几何候选。全量核验：`validate_adaptive.py`；规则单测：`python -m unittest test_candidate_selection test_experiment test_occlusion`；前端单测：`node --test ../pano-web/test_camera_filter.mjs`。

网页：sharp 环境运行 `serve_camera_lab.py --port 8769`。浏览器端 `pano-web/test_camera_lab_browser.js` 核对全部场景的列表、地图分类、导航、固定阈值和导出。旧版 25% 验证记录属于历史，不作为当前规则通过的证据。
