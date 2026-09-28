"""Resumable ranged download of the pinned official DepthMaster checkpoint."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import os
import shutil
import time
import urllib.request


URL = 'https://huggingface.co/VCLab-PolyU/DepthMaster/resolve/add341b64ae9f90372b6483b334ac496a18fe081/depthmaster.pt'
SIZE = 1541004203
SHA256 = 'e61e617acdd0b40bd2c217def5d60b0bd05352e6ae61d940a49c2374c6c75982'
ROOT = Path(__file__).resolve().parent / 'checkpoints'
TARGET = ROOT / 'depthmaster.pt'
PARTS = ROOT / '.depthmaster-download'
CHUNK = 16 * 1024 * 1024


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download_part(index):
    start = index * CHUNK
    end = min(SIZE, start + CHUNK) - 1
    path = PARTS / f'{index:04d}.part'
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return path
    for attempt in range(8):
        offset = path.stat().st_size if path.exists() else 0
        if offset > expected:
            raise RuntimeError(f'Unexpected oversized partial file: {path}')
        if offset == expected:
            return path
        try:
            request = urllib.request.Request(
                URL, headers={'Range': f'bytes={start + offset}-{end}'},
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                wanted = f'bytes {start + offset}-{end}/{SIZE}'
                if response.status != 206 or response.headers.get('Content-Range') != wanted:
                    raise RuntimeError(f'Invalid ranged response for part {index}')
                with path.open('ab') as stream:
                    while data := response.read(1024 * 1024):
                        stream.write(data)
            if path.stat().st_size != expected:
                raise RuntimeError(f'Incomplete part {index}')
            return path
        except Exception as exc:
            print(f'Part {index}: retry {attempt + 1}: {type(exc).__name__}: {exc}', flush=True)
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f'Part {index} failed after retries')


def main():
    ROOT.mkdir(exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size == SIZE and digest(TARGET) == SHA256:
        print('Checkpoint already verified.', flush=True)
        return
    PARTS.mkdir(exist_ok=True)
    count = (SIZE + CHUNK - 1) // CHUNK
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download_part, i) for i in range(count)]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            print(f'Downloaded {done}/{count} parts; elapsed={time.perf_counter() - started:.1f}s', flush=True)
    assembled = ROOT / 'depthmaster.verified-download'
    with assembled.open('wb') as destination:
        for index in range(count):
            with (PARTS / f'{index:04d}.part').open('rb') as source:
                shutil.copyfileobj(source, destination, 4 * 1024 * 1024)
    actual = digest(assembled)
    if assembled.stat().st_size != SIZE or actual != SHA256:
        raise RuntimeError(f'Checkpoint integrity failure: sha256={actual}')
    os.replace(assembled, TARGET)
    print(f'VERIFIED {TARGET} bytes={SIZE} sha256={actual}', flush=True)
    for index in range(count):
        (PARTS / f'{index:04d}.part').unlink()
    PARTS.rmdir()


if __name__ == '__main__':
    main()
