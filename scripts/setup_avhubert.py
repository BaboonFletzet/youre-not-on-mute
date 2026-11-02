"""Fetch pinned AV-HuBERT conversion source and verified official visual checkpoint."""
from pathlib import Path
import hashlib,subprocess,sys,urllib.request
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.setup_accuracy_assets import FILES,digest
REVISION='841efba840404582b5c8e0073b9599cc367e9c5b'
source=ROOT/'.cache/avhubert-s2s'
if not source.exists():subprocess.run(['git','clone','https://github.com/nguyenvulebinh/AV-HuBERT-S2S.git',str(source)],check=True)
subprocess.run(['git','-C',str(source),'checkout',REVISION],check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/setup_accuracy_assets.py'),'avhubert'],check=True)
config=ROOT/'.cache/avhubert-hf-config.txt'
expected='edc4ff41623cf6ff46a697b332871d84eecb6549acc1c9302d8e1af57bd1bead'
if not config.exists():
 data=urllib.request.urlopen('https://huggingface.co/nguyenvulebinh/AV-HuBERT-MuAViC-en/resolve/main/config.json',timeout=30).read()
 if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Architecture template changed; audit before conversion')
 config.write_bytes(data)
if digest(config)!=expected:raise ValueError('Architecture template mismatch')
subprocess.run([sys.executable,str(ROOT/'scripts/convert_avhubert.py')],check=True)
