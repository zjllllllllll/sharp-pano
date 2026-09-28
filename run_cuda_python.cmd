@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "PANO_ROOT=%~dp0"
if exist "%PANO_ROOT%config.local.cmd" call "%PANO_ROOT%config.local.cmd"
if not defined VCVARS64 set "VCVARS64=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not defined CUDA_HOME set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if not exist "%VCVARS64%" (
  echo ERROR: Configure VCVARS64 in config.local.cmd.
  exit /b 2
)
if not exist "%CUDA_HOME%\bin\nvcc.exe" (
  echo ERROR: Configure CUDA_HOME in config.local.cmd.
  exit /b 2
)
call "%VCVARS64%" >nul
if errorlevel 1 exit /b 2
set "PATH=%PANO_ROOT%.envs\runtime\Scripts;%CUDA_HOME%\bin;%PATH%"
if not defined MAX_JOBS set "MAX_JOBS=4"
set "DISTUTILS_USE_SDK=1"
set "TORCH_EXTENSIONS_DIR=%PANO_ROOT%.cache\torch_extensions"
rem Leave TORCH_CUDA_ARCH_LIST unset to target the actual installed GPU.
if "%~1"=="" (
  echo ENVIRONMENT_OK
  exit /b 0
)
call "%PANO_ROOT%run_python.cmd" %*
exit /b %ERRORLEVEL%
