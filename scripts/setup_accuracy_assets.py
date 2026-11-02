"""Pinned, checksum-verified assets for offline accuracy experiments (research use)."""
from __future__ import annotations
import argparse,hashlib,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LRS='https://huggingface.co/datasets/mattymchen/lrs3-test/resolve/9c28ed4b7c87b720f030b5996ef6e7789d7f4a8a/data/train-00000-of-00002-6d235766a16b101a.parquet'
GPT='https://huggingface.co/openai-community/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e/'
FILES={
'dataset':[('.cache/datasets/lrs3-pilot.parquet',LRS,'83ed27da72318d23ac316fcbbdb7834818fe90e095cce036b5c69d3d5a0e71cb')],
'lm':[(f'.cache/models/gpt2/{name}',GPT+name,sha) for name,sha in {
'model.safetensors':'248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707',
'config.json':'0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4',
'tokenizer.json':'8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6',
'vocab.json':'196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783',
'merges.txt':'1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5',
 'tokenizer_config.json':'5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f'}.items()],
'avhubert':[('.cache/models/avhubert-base-vsr.pt','https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/base_vox_433h.pt','84c83698c831b447cb75db0d14b72040c021c9a442ee1208d306b1091c00756d')]}


def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('asset',choices=FILES);p.add_argument('--verify',action='store_true');args=p.parse_args()
 for name,url,sha in FILES[args.asset]:
  path=ROOT/name
  if not path.exists():
   if args.verify:raise SystemExit(f'Missing {path}')
   path.parent.mkdir(parents=True,exist_ok=True);part=path.with_suffix(path.suffix+'.part')
   with urllib.request.urlopen(url,timeout=60) as r,part.open('wb') as f:
    while block:=r.read(1048576):f.write(block)
   if digest(part)!=sha:
    part.unlink();raise SystemExit(f'Checksum mismatch: {name}')
   part.replace(path)
  if digest(path)!=sha:raise SystemExit(f'Existing checksum mismatch: {name}')
  print(f'Verified {name}',flush=True)

if __name__=='__main__':main()
