"""Serial GPU experiments so latency measurements do not compete for the GPU."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PYTHON=str(ROOT/'.venv/bin/python')
manifest='.cache/accuracy/manifest.jsonl'
experiments=[
 ('dev-frozen',manifest,['--variant','frozen','--limit','20']),
 ('dev-jpeg',manifest,['--variant','jpeg','--limit','20']),
 ('dev-stream',manifest,['--streaming','--limit','30']),
 ('dev-overlap',manifest,['--streaming','--overlap','1','--limit','30']),
 ('grid-baseline','.cache/accuracy/grid-manifest.jsonl',['--limit','12']),
 ('grid-vallr','.cache/accuracy/grid-manifest.jsonl',['--model','vallr_stream','--limit','12']),
]
for name,manifest,extra in experiments:
    command=[PYTHON,'-u','scripts/evaluate_accuracy.py',manifest,'--split','dev','--device','mps',
             '--options','.cache/accuracy/beam8-tuned.json','--output',f'.cache/accuracy/{name}.jsonl',*extra]
    with (ROOT/f'.cache/accuracy/{name}.log').open('w') as log:
        print('RUN',name,flush=True)
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print((ROOT/f'.cache/accuracy/{name}.summary.json').read_text(),flush=True)
