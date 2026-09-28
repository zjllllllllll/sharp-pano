"""Keep subprocess compiler/DLL discovery consistent with the project launcher.

Some Windows hosts provide both Path and PATH in their inherited environment;
Python normalizes those names and can otherwise keep a stale value.
"""
import os
from pathlib import Path

_dll_handles = []
if os.name == 'nt' and os.environ.get('PANO_RUNTIME_PATH'):
    os.environ['PATH'] = os.environ['PANO_RUNTIME_PATH']
    directory = Path(__file__).resolve().parents[1]/'.envs/runtime/Library/bin'
    if directory.is_dir():
        _dll_handles.append(os.add_dll_directory(str(directory)))
