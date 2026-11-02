"""Small development-only crop/scaling sweep; report phoneme errors before word decoding."""
import json,os,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import cmudict
from services.inference.vallr import VALLRRecognizer
from services.inference.evaluation import edits
from scripts.evaluate_accuracy import read_frames
manifest=ROOT/'.cache/accuracy/grid-manifest.jsonl'
rows=[r for r in map(json.loads,manifest.read_text().splitlines()) if r['split']=='dev'][:3]
model=VALLRRecognizer('mps');dictionary=cmudict.dict();results=[]
try:
 for row in rows:
  frames,_=read_frames(row['video'])
  expected=' '.join(re.sub(r'\d','',p) for word in row['reference'].split() for p in dictionary[word][0])
  for crop in (False,True):
   for scale in ('raw','unit','symmetric'):
    result=model.recognize_stream(frames,crop=crop,scale=scale)
    counts=edits(expected,result.get('phonemes',''))
    error=sum(counts[k] for k in ('substitutions','deletions','insertions'))/max(1,counts['reference_words'])
    item={'id':row['id'],'crop':crop,'scale':scale,'phoneme_error_rate':error,'expected':expected,**result}
    results.append(item);print(json.dumps(item),flush=True)
finally:model.close()
(ROOT/'.cache/accuracy/vallr-input-audit.json').write_text(json.dumps(results,indent=2))
