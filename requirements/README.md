# Python dependency files

Supported installation baseline: Windows x64, Python 3.13, NVIDIA CUDA Toolkit
12.8, Visual Studio 2022 Build Tools with C++ x64 tools, Conda, and Node.js/npm.

Run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install.ps1
```

The installer uses these files in order:

1. `cuda128.txt`: PyTorch 2.8.0 and torchvision 0.23.0 from the CUDA 12.8 index.
2. `windows-py313.lock.txt`: pinned runtime packages from PyPI, including gsplat.
3. Editable, dependency-free registrations of the bundled SHARP and utils3d
   sources. Their dependencies are supplied by the two files above.
4. `scripts/patch_runtime.py`: local Windows compiler compatibility changes.
5. `pano-web/package-lock.json`: frontend packages installed with `npm ci`.

These are Windows runtime pins, not a cross-platform lockfile. Installation
also resolves build-system dependencies declared by source distributions and
editable packages. Linux/macOS installation is not verified.

The environment lives in `.envs/runtime`; installed packages are excluded from
Git. CUDA, the compiler, Conda and Node.js remain system prerequisites.
Do not combine the two pip commands into one command: the CUDA file selects a
separate package index.

To install the environment before downloading models:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install.ps1 -SkipModels
.\run_python.cmd scripts/download_models.py
.\run_cuda_python.cmd scripts/check_install.py --gpu
```

The install check imports modules and loads/compiles the gsplat CUDA extension.
It does not run scene inference or call a paid image service. The three models
are described in [models/README.md](../models/README.md).
