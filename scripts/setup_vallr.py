"""Fetch VALLR's pinned source/checkpoint and verify the checkpoint before use."""
from __future__ import annotations

import hashlib
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = '9793489136bc00293242a5410def05ba2806e16f'
FILE_ID = '14u7MRTxXL1psHMnlssw_Drq-zRMGC7zD'
SHA256 = '967667d61e705b0bc78d40a3ec80dd7ec449aa1d4d90a5e83c0d4289309ef374'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download_checkpoint(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + '.part')
    initial_url = f'https://drive.google.com/uc?export=download&id={FILE_ID}'
    request = urllib.request.Request(initial_url, headers={'User-Agent': 'YoureNotOnMute/0.2'})
    with urllib.request.urlopen(request, timeout=30) as response:
        content_type = response.headers.get('Content-Type', '')
        first = response.read(4096)
        if 'text/html' in content_type or first.lstrip().startswith(b'<!DOCTYPE html'):
            html = first.decode('utf-8', errors='replace')
            if 'uuid' not in html:
                raise RuntimeError('Google Drive returned an unexpected download page.')
            uuid = re.search(r'name="uuid" value="([^"]+)"', html)
            if not uuid:
                raise RuntimeError('Could not find Google Drive confirmation token.')
            query = urllib.parse.urlencode({'id': FILE_ID, 'export': 'download',
                                            'confirm': 't', 'uuid': uuid.group(1)})
            download_url = f'https://drive.usercontent.google.com/download?{query}'
            response.close()
            request = urllib.request.Request(download_url, headers={'User-Agent': 'YoureNotOnMute/0.2'})
            with urllib.request.urlopen(request, timeout=60) as download:
                with part.open('wb') as out:
                    while chunk := download.read(1024 * 1024):
                        out.write(chunk)
        else:
            with part.open('wb') as out:
                out.write(first)
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
    if digest(part) != SHA256:
        part.unlink(missing_ok=True)
        raise RuntimeError('VALLR checkpoint SHA-256 mismatch; refusing to install it.')
    part.replace(target)


def main():
    source = ROOT / '.cache' / 'vallr'
    if not source.exists():
        subprocess.run(['git', 'clone', 'https://github.com/MarshallT-99/VALLR.git', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'checkout', REVISION], check=True)
    checkpoint = ROOT / '.cache' / 'models' / 'vallr.path'
    if not checkpoint.exists():
        print('Downloading official VALLR checkpoint (696 MB)…', flush=True)
        download_checkpoint(checkpoint)
    if digest(checkpoint) != SHA256:
        raise SystemExit('Existing VALLR checkpoint checksum mismatch.')
    print(f'Verified VALLR source {REVISION}; checkpoint SHA-256 {SHA256}')


if __name__ == '__main__':
    main()
