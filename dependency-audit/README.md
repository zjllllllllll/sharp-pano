# 开源前依赖检查

> 历史快照（2026-09-24）：以下路径和结论描述迁移前状态。2026-09-28 已将主流程依赖迁入仓库，并提供公司 / GrsAI 配置；当前安装和目录说明请看 [根目录 README](../README.md)，第三方来源看 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)。本报告保留用于比较。

检查日期：2026-09-24。检查对象为当前 `pano-web` 主流程（初始建模、相机与 Mask、RGB 修补、LingBot 深度、SHARP 补丁、第二轮），以及它直接引用的本地代码。另检查了主要实验目录是否被主流程引用。

这是源码、文件存在性、安装包元数据和 Git 状态检查；没有调用图片 API、运行 GPU 推理或修改业务代码。`snapshot.json` 保存了关键外部文件哈希、模型大小、主要 Python 包版本和两份 SHARP 的比较结果；它不是完整的环境锁文件或全目录安全扫描。

## 结论

**当前复制 `D:/sharp-pano` 还不能独立运行完整流程；只复制 `pano-web` 更不够。**

项目仍直接读取 `D:/gauss-hole` 下两个工程的 Python 文件、模型权重和启动脚本，并使用用户目录里的三个 Conda 环境。当前根目录尚未建立 Git 仓库，也没有统一的根级安装说明、依赖锁定、忽略规则和项目许可证。

可以整理为一个根目录：项目代码、必要第三方源码、补丁、环境定义、模型下载脚本均放入仓库；安装后的环境和权重也可以位于该根目录下，但由安装脚本创建并加入 Git 忽略规则。CUDA、显卡驱动、Windows 编译器仍是系统依赖，在线图片修补仍需用户配置服务和凭据。

## 1. 当前目录之外的直接代码依赖

| 功能 | 当前外部位置 | 本项目调用位置 | 整理时需要包含的内容 |
|---|---|---|---|
| GPT 图片修补 | `D:/gauss-hole/跑通补洞/studio/` | `pano-web/gpt_green_repair.py:125`；REFERENCE 来自 `gpt_rgb_experiment.py:19` | `image_api.py`、`errors.py`、包入口；应作为本项目的图片服务适配层固定下来 |
| Seedream 图片修补 | `D:/gauss-hole/weizhuang/sharp_sd2_test-master/click_gaussian/` | `pano-web/novel_pipeline.py:23`、`:262` | `image_repair.py`、`ark_credentials.py`、包入口；运行时还依赖 requests/Pillow/NumPy/OpenCV/SciPy |
| 深度校准和补全辅助函数 | `D:/gauss-hole/跑通补洞/fuse_and_validate_depth.py` | `pano-web/novel_depth_geometry.py:43`、`novel_gaussian_geometry.py:19` | 当前版本的辅助函数及来源说明 |
| 深度锚定 SHARP 补丁 | 同一外部工程 | `pano-web/novel_gaussian.py:20`、`:21`、`:72` | `run_sharp_hard_depth.py`、`sharp_runtime.py` |
| 补丁变换、透明度融合、渲染和核验 | 同一外部工程 | `pano-web/novel_gaussian_geometry.py:59`；`novel_gaussian.py:123`、`:224` | `merge_hard_patch.py`、`gaussian_patch_io.py` |
| 补洞阶段的 SHARP 本体 | `D:/gauss-hole/跑通补洞/third_party/ml-sharp/` | `pano-web/novel_gaussian.py:113`，以及外部 `sharp_runtime.py` | 模型源码、gsplat 兼容修改、源码/模型许可证 |
| LingBot-Depth 本体 | `D:/gauss-hole/跑通补洞/third_party/lingbot-depth/` | `pano-web/novel_depth.py:17`；`quality_depth.py:21`、`:58` | `mdm` 模型代码、配置、Windows fallback、许可证 |
| GPU 启动器 | `D:/gauss-hole/跑通补洞/run_cuda_python.cmd` | `pano-web/server.py:80`、`novel_depth.py:18` | 移入项目并让路径、GPU 架构和环境可配置 |
| Seedream 启动器 | `D:/gauss-hole/weizhuang/sharp_sd2_test-master/run_goris_cuda_python.cmd` | `pano-web/novel_pipeline.py:24`；`server.py:81` | **检查时文件不存在，当前引用需要修复** |

不需要把外部工程的整个 Viewer、训练工具和历史输出一起搬入；上述调用链才是当前工作台所用部分。相关代码的分发归属也需要记录，不能只搬文件而丢失原许可。

## 2. 已在根目录内，但不在 pano-web 内的必要代码

| 目录 | 当前用途 | 结论 |
|---|---|---|
| `DepthMaster/` | 第 2 步原始全景深度；入口 `infer_panoramas.py` | 必须保留模型和本地推理适配代码 |
| `DepthMaster/third_party/utils3d/` | DepthMaster 的几何工具 | 已经在目录内，但由外部 depthmaster 环境以 editable 方式安装；换路径后要重新安装 |
| `ml-sharp/` | 第 3～5 步初始化和调整高斯 | 必须保留；与补洞所用的外部 SHARP 目前不是完全相同的源码 |
| `sharp-sixface/` | 六面投影、SHARP 加载、PLY/高斯转换 | 主流程直接导入，并非纯实验目录 |
| `room-outline-experiment/` | 房间轮廓、候选相机、自由空间与 Mask | 主流程直接导入 `batch`、`outline`、`cameras`、`candidate_selection`、`occlusion` 等；不能按目录名判断为可删除实验 |
| `room-outline-experiment/.deps/` | Shapely 2.1.2 | 当前是就地安装的二进制包，应改成安装依赖 |
| `room-outline-experiment/.render-deps/` | embreex 4.4.0，射线求交 | 同上；`render-requirements.txt` 只记录了这一项，不是整个应用的依赖清单 |
| `pano-web/node_modules/` | Three.js 0.180.0、Spark 2.2.0 | 已有 `package.json`/`package-lock.json`，静态服务从本地提供；可由 `npm ci` 恢复 |

当前主流程没有发现对 `UniSHARP/`、其 `UniK3D/`、`depth-benchmark/`、根目录 `repos/`、`tools/convert_ply_to_las.py` 的调用。它们可以按独立实验处理，不必进入当前应用的最小发布包。

`pano-tests/` 是离线批处理样本来源；上传新图的主流程不需要这些历史样本。相机实验栏目默认读取 `room-outline-experiment/runs/20260922-adaptive-sampling`，未附带数据时该栏目会显示空目录提示；新任务的相机计算代码仍需保留。

## 3. 源码版本和本地修改

| 来源 | 检查到的版本 | 必须保留的本地内容 |
|---|---|---|
| 根目录内 SHARP | `1eaa046834b81852261262b41b0919f5c1efdd2e` | 跟踪源码无本地修改 |
| DepthMaster | `c2ec897fa14803f05f211f6de49571801a4edfa1` | 已修改 `depthmaster/model/mirror_vgt/layers/attention.py` 和 `eval_panorama/eval.py`；本地新增 `infer_panoramas.py`、`download_checkpoint.py` 等尚非上游跟踪文件 |
| DepthMaster 的 utils3d | `3fab839f0be9931dac7c8488eb0e1600c236e183` | 应固定该来源提交，不能只写未锁定的 `utils3d` 包名 |
| 跑通补洞工程 | `df891c6ea2182ba4f1301b292bc55ed7f07549bc` 加当前工作区修改 | `studio/image_api.py` 有未提交修改，因此仅记录 HEAD 无法还原当前适配器 |
| 旧 Seedream 工程 | `3f91c91a0bd27d4ce8f2c1d599660e6ee7c35302` | 本项目引用的启动器当前缺失 |

逐文件比较两份 SHARP 的全部 42 个 Python 源文件（忽略换行格式）：仅 `sharp/utils/gsplat.py` 文本不同。外部版本适配 gsplat 返回 3 项或 4 项结果。整理时可以考虑合并成一份，但必须保留并验证这一差异，不能直接用上游原文件替换。

外部 LingBot 的声明记录了 `mdm/model/dinov2_rgbd/layers/block.py` 的无 xFormers Windows fallback。SHARP/LingBot 在外部工程中是随仓库保存的源码，不是独立 Git 子仓库；对其目录执行 `git rev-parse HEAD` 会得到外层工程的提交，不能误记为它们各自的上游版本。

## 4. Python 和系统环境

`pano-web/start.cmd` 与 `server.py:31` 默认使用 `%LOCALAPPDATA%/miniforge3/envs/`；外部 GPU 启动器使用 `%USERPROFILE%/AppData/Local/miniforge3/envs/goris`，允许用本地配置覆盖。

| 环境 | 用途 | 检查到的关键版本 |
|---|---|---|
| sharp | 服务、初始 SHARP、点云处理 | torch 2.8.0+cu128、torchvision 0.23.0+cu128、NumPy 2.5.2、SciPy 1.16.2 |
| depthmaster | 原始深度、相机候选与 Mask 工作进程 | torch 2.8.0+cu128、torchvision 0.23.0+cu128、OpenCV 4.11.0.86、utils3d 1.3（上述本地提交） |
| goris | 后续深度渲染、LingBot、SHARP 补洞；当前 GPT 分支也经过该启动器 | torch 2.11.0+cu128、torchvision 0.26.0+cu128、gsplat 1.5.3、NumPy 2.2.6、SciPy 1.15.3 |

其他主链路依赖包括 Pillow、NumPy、SciPy、OpenCV、plyfile、laspy、timm、einops、huggingface_hub、safetensors、Shapely、embreex 等。完整选定包的已安装版本见 `snapshot.json`；传递依赖仍需在整理时生成和验证锁文件。

根目录内 `.envs/unisharp` 不是网页主流程默认使用的环境。把它提交或移动并不能替代上述三个环境。也不能直接把当前三套包拼成一套；torch、NumPy 等存在版本差异，需要先做兼容验证。可先让三套环境由脚本创建在同一个项目根目录下。

外部 GPU 启动器还明确依赖 CUDA Toolkit 12.8、Visual Studio 2022 Build Tools、Ninja，并把 `TORCH_CUDA_ARCH_LIST` 写死为 `12.0`。这不适合不经配置直接分发给不同 GPU 的使用者。Windows 字体 `C:/Windows/Fonts/msyh.ttc` 也被相机实验绘图脚本硬编码。

goris 的 torch/torchvision 安装元数据保留了旧本机 wheel 文件路径。这是安装来源记录，不代表运行时还会读取该 wheel；复现安装时应改成可获得的版本来源，不能把旧路径原样写进公开安装文件。

## 5. 模型权重

| 模型 | 当前位置 | 实际文件大小 |
|---|---|---:|
| DepthMaster | `DepthMaster/checkpoints/depthmaster.pt` | 1,541,004,203 B |
| 初始 SHARP | `ml-sharp/checkpoints/sharp_2572gikvuh.pt` | 2,809,738,232 B |
| 补洞 SHARP | 外部 `third_party/ml-sharp/ckpt/sharp_2572gikvuh.pt` | 2,809,738,232 B |
| LingBot | 外部 `third_party/lingbot-depth/model/lingbot-depth/model.pt` | 1,284,837,952 B |

两份 SHARP 文件的已有项目文档记录相同 SHA256；本次只核对了文件存在性和大小，没有重新哈希全部大模型。按去重后的三种模型计算约 5.64 GB，当前四个文件共约 8.45 GB。

建议在根目录统一使用 `models/`，通过下载脚本按版本和 SHA256 验证，模型文件在安装后仍位于同一个项目文件夹。当前只有 DepthMaster 目录有明确的本地下载脚本，尚无覆盖整条流程的统一下载入口。

## 6. 已发现的实际失配

1. **Seedream 启动器不存在。** `novel_pipeline.LAUNCHER` 指向的 `run_goris_cuda_python.cmd` 不存在；按 `server.execute_stage` 当前分支，新启动 Seedream 工作进程会失败。
2. **GPT 适配器的服务地址已经漂移。** `gpt_green_repair.py:129` 只接受 `oneapi.qunhequnhe.com`，但外部 `studio/image_api.py` 当前把 `gpt-image-2` 指向 `grsai.dakka.com.cn`。新的 GPT 修补请求会被本项目的 preflight 拒绝；历史缓存仍可能展示正常。本次没有改地址或发送请求。
3. **健康检查覆盖不完整。** `server.py:352` 当前只检查初始 DepthMaster/SHARP 的 Python 和权重，不能发现上述补洞依赖缺失。
4. **当前安装说明不完整。** `pano-web/README.md` 还使用旧阶段描述，Python 包安装、外部辅助代码、模型准备和后续轮次未统一成新机器可复现入口。

## 7. 仓库边界、凭据和第三方声明

- 根目录目前不是 Git 仓库；DepthMaster、ml-sharp、utils3d 等有自己的嵌套 `.git`。整理时应明确采用随仓库保存的源码或固定子模块，并保存本地补丁，避免只提交嵌套仓库指针而漏掉适配代码。
- `pano-web/.gitignore` 仅覆盖该子目录的 `node_modules/`、`runs/`、`.runtime/`、`__pycache__/`。根级环境、缓存、权重、其他 runs、日志和诊断产物没有统一忽略规则。
- 图片接口从环境变量或外部工程的 key 文件取凭据。迁移的是适配器和配置模板；凭据文件应排除。本次只记录相关文件是否存在，没有读取密钥值，亦未做全目录秘密扫描。
- 当前根目录无项目级 LICENSE。DepthMaster 附 MIT，LingBot 附 Apache-2.0，SHARP 有独立 `LICENSE` 和 `LICENSE_MODEL`。本地 `LICENSE_MODEL` 明确区分科研用途并排除商业利用/产品用途；不能把整套代码与权重统一描述成任意用途的 MIT 项目。发布材料应分别保留原文件和第三方声明。

## 8. 可整理成的单一根目录

```text
sharp-pano/
  README.md
  LICENSE                         自有代码的许可证，尚待所有者选择
  THIRD_PARTY_NOTICES.md
  .gitignore
  pano-web/                       应用、前端和 npm 锁文件
  sharp-sixface/                  本项目六面适配
  geometry/                       从实验目录提取主流程需要的几何模块
  repair/                         固定下来的深度/SHARP 补洞适配代码
  providers/                      GPT/Seedream 接口适配与配置
  third_party/
    DepthMaster/
    ml-sharp/
    lingbot-depth/
    utils3d/
  requirements/                   按环境锁定依赖
  scripts/                        安装、下载模型、启动、健康检查
  config.example.*
  models/                         安装后下载到这里，Git 忽略
  .envs/                          安装后创建到这里，Git 忽略
  runs/                           运行时生成，Git 忽略
  experiments/                    可选历史实验
```

上面是待实施布局，不是已经搬迁的结果。下一步应按顺序完成：保存必要源码和本地补丁 → 消除外部绝对路径并固定接口配置 → 整理环境定义和权重下载 → 在独立路径的新安装中验证一次完整流程。此次仅新增检查报告和快照，未改动前端、运行数据、环境或算法。
