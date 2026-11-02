"""Paired utterance bootstrap for two runs on the same evaluation manifest."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from services.inference.evaluation import edits,summarize
p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
a={r['id']:r for r in map(json.loads,args.baseline.read_text().splitlines())}
b={r['id']:r for r in map(json.loads,args.candidate.read_text().splitlines())}
if a.keys()!=b.keys():raise ValueError('Evaluation clip sets differ')
counts=[]
for key in sorted(a):
 if a[key]['reference']!=b[key]['reference']:raise ValueError('Reference mismatch')
 x,y=edits(a[key]['reference'],a[key]['text']),edits(b[key]['reference'],b[key]['text'])
 counts.append([x['reference_words'],sum(x[k] for k in ('substitutions','deletions','insertions')),sum(y[k] for k in ('substitutions','deletions','insertions'))])
counts=np.array(counts);rng=np.random.default_rng(29);deltas=[]
for _ in range(2000):
 sample=counts[rng.integers(0,len(counts),len(counts))].sum(0)
 deltas.append(float((sample[1]-sample[2])/max(1,sample[0])))
report={'baseline':summarize(list(a.values())),'candidate':summarize(list(b.values())),
        'wer_reduction_95_percent_interval':np.percentile(deltas,[2.5,97.5]).tolist(),
        'note':'Utterance bootstrap, not speaker bootstrap: source lacks speaker identifiers.'}
if args.output:args.output.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
