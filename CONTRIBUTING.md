# Contributing

安装和启动步骤见 [README](README.md)。问题报告和贡献可以使用中文或英文。

## 开发与验证

使用项目内的 `.envs/runtime`，从仓库根目录运行：

```powershell
.\run_python.cmd -B -m unittest discover -s scripts -p "test_*.py"
.\run_cuda_python.cmd -B -m unittest discover -s pano-web -p "test_*.py"
.\run_python.cmd -B -m unittest discover -s sharp-sixface -p "test_*.py"
.\run_python.cmd -B -m unittest discover -s room-outline-experiment -p "test_*.py"
.\run_python.cmd -B -m unittest discover -s providers/studio/tests -p "test_*.py"
node --test pano-web/test_camera_filter.mjs pano-web/test_point_filter.mjs pano-web/test_image_wipe.mjs pano-web/test_view_cache.mjs
.\run_python.cmd -B scripts/check_release.py
```

单元测试使用合成数据和模拟服务；主流程测试导入修补模块时会加载 gsplat CUDA 扩展，因此需 CUDA 启动器。实际 GPU 推理、浏览器操作与图片服务调用应在说明中单独记录。
只运行与修改有关的测试；提交 PR 时写明问题、变化和实际验证结果。

## 代码位置

- `pano-web/`：网页服务和流水线调度。
- `sharp-sixface/`：投影、几何、高斯初始化和 SHARP 适配。
- `room-outline-experiment/`：主流程共用的轮廓、相机与未知空间计算。
- `repair/`、`providers/`：修补算法和图片服务适配器。
- `third_party/`：固定的第三方源码，保留其许可证和来源。

主流程使用根目录的 `models/` 和 `third_party/`，不依赖其他本地工程。
修改第三方源码时，在 `THIRD_PARTY_NOTICES.md` 描述修改，并同步
`third_party/SOURCE_SNAPSHOT.json` 中对应文件的 SHA-256（按 `hash_mode` 处理换行）。
`original_migration_hashes` 是历史来源记录，不要覆盖。

权重、运行环境、密钥、上传的场景和运行结果不进入源码提交。
自有代码的贡献按 [MIT License](LICENSE) 分发；第三方代码仍遵循各自许可证。
