# SHARP Panorama Workbench

本地全景 RGB → DepthMaster 深度 → SHARP 高斯 → 相机与 Mask → 图片修补 → LingBot 深度补全 → 深度锚定 SHARP 补洞。
当前网页包含首次建模的 5 步，以及每轮 4 步的循环修补，最多修补 5 轮。SHARP 单面生成与最终补丁渲染只记录诊断，不再按深度误差、alpha、中心误差或覆盖率拒收和阻止下一轮。

应用运行所用源码、Python 环境、前端包、模型和 CUDA 扩展均位于本项目目录内；不需要其他本地工程。显卡驱动、CUDA Toolkit、系统编译器和安装用的 Conda/Node.js 是系统前置条件。图片修补调用所选的在线服务。

## 迭代停止规则

每轮高斯补丁合并后，用初始观测与所有已接受轮次更新累计空间，重算固定候选位置的几何 Mask。停止统计只包含剩余未处理的可选相机；排除初始位置、已处理位置和既有流程标记失败的位置。

占比统一为清理后的几何 Mask 像素数 / 全景总像素数，不使用球面加权，也不混入渲染 alpha Mask。标准差使用总体标准差（`ddof=0`），单位为百分点；判断使用未四舍五入的数值。

满足任意一项即停止：

- 剩余视角的最大占比 ≤5%。
- 最大占比 ≤7%，且标准差 ≤1.5 个百分点。
- 已完成 5 轮修补（首次建模不计入轮数）。

候选耗尽时单独记录为“没有剩余候选”，不宣称面积达标。评估位于下一轮选相机、截图和付费修补之前；报告与逐相机像素占比保存在主任务的 `iteration-evaluations/`，服务与页面共同使用该停止状态。

## Windows 安装

```powershell
git clone https://github.com/zjllllllllll/sharp-pano.git
cd sharp-pano
```


基线：Windows x64、Python 3.13、PyTorch 2.8.0 + CUDA 12.8、gsplat 1.5.3。
准备 NVIDIA 驱动、CUDA Toolkit 12.8、Visual Studio 2022 Build Tools（C++ x64）、Conda 和 Node.js。

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install.ps1
```

脚本创建 `.envs/runtime`，安装锁定依赖和仓库内源码，恢复 npm 包，下载并校验三种模型，最后检查模块加载来源和 CUDA。可用 `-Conda <conda.exe路径>` 指定 Conda；`-SkipModels` 跳过权重下载和权重存在性检查，仍验证运行环境与 CUDA。依赖安装顺序见 [requirements/README.md](requirements/README.md)。
模型总计约 5.64 GB；运行环境、编译缓存和任务结果另需磁盘空间。首次 gsplat 编译可能需要数分钟。

若 CUDA/VS 安装位置不同，**安装前**把 `config.example.cmd` 复制成 `config.local.cmd` 后修改。GPU 架构默认自动检测，不固定为某一张显卡。当前验证平台为 Windows；Linux/macOS 没有完整安装验证。

单独下载或验证权重：

```powershell
.\run_python.cmd scripts/download_models.py
.\run_python.cmd scripts/download_models.py --verify-only
```

固定链接、SHA-256、大小和断点续传说明见 [models/README.md](models/README.md)。

## 图片服务配置

两套 GPT 接口分别保留，通过根目录 `config.local.json` 选择。新建任务时固定该选择，后续轮次继承同一配置；已有任务不会因全局配置切换而更换服务。

公开默认配置为 GrsAI：

```json
{"image_provider": "grsai"}
```

将密钥放到 `secrets/grs-key.txt`，或设置 `GRSAI_API_KEY`。适配器：`providers/studio/image_api.py`。

公司接口用于本地测试：

```json
{"image_provider": "company"}
```

将公司接口密钥放到 `secrets/api_key.txt`，或设置 `CLICKGS_IMAGE_API_KEY`。本地密钥文件优先于环境变量，便于替换旧的系统配置。GPT 图片输入还需要 OSS：在 `secrets/company-oss.json` 中配置 `OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET`、可选的 `OSS_SESSION_TOKEN`，或设置同名环境变量。适配器：`providers/studio/company_image_api.py` 和 `oss_input.py`。`config.example.json` 使用公开默认 GrsAI。公司接口需要相应账号和访问权限，普通安装可使用 GrsAI。

`PANO_IMAGE_PROVIDER` 可覆盖新任务的默认提供方。凭据、`config.local.*` 和整个 `secrets/` 目录均被 Git 忽略。不要在截图、问题报告或日志中贴密钥。

历史 Seedream 分支的必要适配代码也已迁入 `providers/click_gaussian`；凭据读取 `secrets/ark-key.txt` 或 `ARK_API_KEY`。

## 启动与检查

双击 `pano-web/start.cmd`，或执行：

```powershell
.\run_python.cmd pano-web/server.py --port 8765
```

访问 `http://127.0.0.1:8765`，上传 2:1 全景图。服务器仅绑定本机。

```powershell
.\run_cuda_python.cmd scripts/check_install.py --gpu --report .runtime/installation-check.json
.\run_python.cmd scripts/download_models.py --verify-only
.\run_python.cmd scripts/check_release.py --report .runtime/release-check.json
.\run_cuda_python.cmd -m unittest discover -s pano-web -p "test_*.py"
node --test pano-web/test_camera_filter.mjs pano-web/test_point_filter.mjs pano-web/test_image_wipe.mjs pano-web/test_view_cache.mjs
```

`check_install.py` 验证必要模块实际从本目录加载、权重存在、图片服务本地配置和 CUDA 扩展可用。权重完整性由 `download_models.py --verify-only` 验证。检查不发送收费图片请求。

## 目录

- `pano-web/`：网页、服务、模型调度、质量判断与迭代。
- `sharp-sixface/`：全景/六面投影和初始高斯适配。
- `room-outline-experiment/`：当前主流程使用的轮廓、相机和自由空间代码；目录名虽含 experiment，但不能整目录删除。
- `repair/`：深度锚定、补丁融合与渲染代码。
- `providers/`：公司、GrsAI 和 Seedream 客户端。
- `third_party/`：固定的 DepthMaster、SHARP、LingBot 和 utils3d 源码，包含本地兼容修改。
- `requirements/`、`scripts/`：环境锁定、安装、模型下载和验证。
- `.envs/runtime/`、`models/`、`.cache/torch_extensions/`：本地安装产物，Git 忽略。
- `pano-web/runs/`：任务输入、输出与日志，Git 忽略。

当前生产路径使用 `third_party/` 和 `models/`。UniSHARP 与四套历史深度 benchmark 已移除；根目录可能保留的旧模型检出、私人测试图和运行产物均被 Git 忽略，不参与安装。`dependency-audit/` 和部分实验文档保留历史设计记录，运行命令以本页为准。

算法细节见 [pano-web/README.md](pano-web/README.md)、[NOVEL_VIEWS.md](pano-web/NOVEL_VIEWS.md) 和 [FACE_QUALITY.md](pano-web/FACE_QUALITY.md)；旧文档内的外部路径属于迁移前记录，运行命令以本页为准。

## 第三方与发布

见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。保留源码和模型各自的许可证，尤其 SHARP 的模型许可。
模型、运行环境、凭据和场景输出不直接提交到 GitHub；源码克隆后由安装脚本在同一目录中恢复依赖。
自有应用代码采用 [MIT License](LICENSE)，第三方源码和模型继续遵守各自许可证。SHARP 权重仅限其模型许可证定义的研究用途，MIT 不改变这项限制。

贡献和测试见 [CONTRIBUTING.md](CONTRIBUTING.md)，安全报告见 [SECURITY.md](SECURITY.md)。
