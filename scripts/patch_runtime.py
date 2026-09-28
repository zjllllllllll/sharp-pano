"""Apply the documented Windows gsplat compiler fix to this local environment."""
from pathlib import Path
import os
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def replace_file(path, content):
    # The migration can use hardlinks. Replace the directory entry so the old
    # environment is never modified when patching this one.
    temporary = path.with_name(path.name+'.pano-tmp')
    temporary.write_text(content, encoding='utf-8')
    os.replace(temporary, path)


def main():
    if sys.platform != 'win32':
        return
    site = ROOT/'.envs/runtime/Lib/site-packages'
    path = site/'gsplat/cuda/_backend.py'
    text = path.read_text(encoding='utf-8')
    before = 'extra_cflags = [opt_level, "-Wno-attributes"]'
    after = 'extra_cflags = (["/Od" if FAST_COMPILE else "/O2"] if os.name == "nt" else [opt_level, "-Wno-attributes"])'
    if before in text:
        replace_file(path, text.replace(before, after))
    elif after not in text:
        raise RuntimeError('Unsupported gsplat compiler configuration; inspect before patching')
    path = site/'torch/utils/cpp_extension.py'
    text = path.read_text(encoding='utf-8')
    before = "SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()"
    if before in text:
        replace_file(path, text.replace(before, "SUBPROCESS_DECODE_ARGS = ('utf-8', 'replace') if IS_WINDOWS else ()"))
    path = site/'torch/include/c10/cuda/CUDACachingAllocator.h'
    text = path.read_text(encoding='utf-8')
    if re.search(r'\bsmall\b', text):
        replace_file(path, re.sub(r'\bsmall\b', 'small_pool', text))
    print('Project-local Windows compiler compatibility patches ready')


if __name__ == '__main__':
    main()
