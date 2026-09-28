@echo off
cd /d "%~dp0.."
call run_python.cmd pano-web/launch.py
pause
