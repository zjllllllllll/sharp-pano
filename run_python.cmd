@echo off
setlocal EnableExtensions DisableDelayedExpansion
set "PANO_ROOT=%~dp0"
set "PANO_ENV=%PANO_ROOT%.envs\runtime"
if not exist "%PANO_ENV%\python.exe" (
  echo ERROR: Install the project-local runtime first. See README.md.
  exit /b 2
)
set "PATH=%PANO_ENV%;%PANO_ENV%\Scripts;%PANO_ENV%\Library\bin;%PATH%"
set "PANO_RUNTIME_PATH=%PATH%"
set "PYTHONPATH=%PANO_ROOT%pano-web;%PANO_ROOT%repair;%PANO_ROOT%providers;%PANO_ROOT%third_party\ml-sharp\src;%PANO_ROOT%third_party\DepthMaster;%PANO_ROOT%third_party\DepthMaster\third_party\utils3d;%PANO_ROOT%third_party\lingbot-depth;%PANO_ROOT%sharp-sixface;%PANO_ROOT%room-outline-experiment"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "HF_HUB_OFFLINE=1"
set "TORCH_EXTENSIONS_DIR=%PANO_ROOT%.cache\torch_extensions"
"%PANO_ENV%\python.exe" %*
exit /b %ERRORLEVEL%
