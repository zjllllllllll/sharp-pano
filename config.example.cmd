@rem Optional machine-specific compiler configuration; copy to config.local.cmd.
@rem Paths below are system prerequisites, not other project checkouts.
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "VCVARS64=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
@rem Optional; otherwise PyTorch detects the installed GPU architecture.
@rem set "TORCH_CUDA_ARCH_LIST=12.0"
set "MAX_JOBS=4"
