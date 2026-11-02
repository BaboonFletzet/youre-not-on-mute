"""Fetch the pinned authors' source and checkpoint; verify before loading."""
import hashlib
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = '182b62837773ab01052d4ac21ef1d2203ea7d267'
SHA256 = 'fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb'
URL = 'https://drive.usercontent.google.com/download?id=1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_&export=download&confirm=t'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    source = ROOT / '.cache/auto-avsr'
    if not source.exists():
        subprocess.run(['git', 'clone', 'https://github.com/mpc001/auto_avsr.git', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'checkout', REVISION], check=True)
    target = ROOT / '.cache/models/autoavsr-en.pth'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        part = target.with_suffix('.pth.part')
        print('Downloading English visual-only checkpoint (~956 MiB)…', flush=True)
        urllib.request.urlretrieve(URL, part)
        if digest(part) != SHA256:
            raise SystemExit('Checkpoint checksum mismatch. Model will not be loaded.')
        part.rename(target)
    if digest(target) != SHA256:
        raise SystemExit('Existing checkpoint checksum mismatch.')
    print(f'Verified source {REVISION} and checkpoint SHA256 {SHA256}')


if __name__ == '__main__':
    main()
