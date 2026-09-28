# 模型权重

模型二进制不随源码提交。安装脚本默认下载三份权重，也可以在仓库根目录单独运行：

```powershell
.\run_python.cmd scripts/download_models.py
.\run_python.cmd scripts/download_models.py --verify-only
```

下载脚本仅依赖 Python 标准库（Python 3.11+），尚未安装项目环境时也可以执行
`python scripts/download_models.py`。运行模型本身仍需根目录 README 中的完整环境。

| 模型 | 保存位置 | 字节数 | 下载来源 |
|---|---|---:|---|
| DepthMaster | `models/depthmaster.pt` | 1,541,004,203 | [Hugging Face](https://huggingface.co/VCLab-PolyU/DepthMaster) |
| SHARP | `models/sharp_2572gikvuh.pt` | 2,809,738,232 | [Apple 官方仓库](https://github.com/apple/ml-sharp) |
| LingBot-Depth v0.5 | `models/lingbot-depth.pt` | 1,284,837,952 | [Hugging Face](https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5) |

合计 **5,635,580,387 字节（约 5.64 GB / 5.25 GiB）**。
初始建模与后续修补共用同一份 SHARP 权重。

[manifest.json](manifest.json) 固定下载 URL、文件大小和 SHA-256；
两个 Hugging Face 链接固定到提交版本。下载写入 `.partial` 文件，支持服务端
Range 断点续传，完成后校验大小与 SHA-256，再发布为正式文件。已有正确权重会跳过，
已有错误权重会保留并报错；确认不需要后移走错误文件再重试。

权重许可证独立于应用 MIT 许可证。尤其 SHARP 权重遵循
[Apple Machine Learning Research Model License](../third_party/ml-sharp/LICENSE_MODEL)，
限其条款定义的研究用途。完整来源和声明见
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。
