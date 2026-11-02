"""Development-only risk/coverage audit; never treats a decoder margin as a probability."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from services.inference.decoding import DecodeOptions,rank_candidates
from services.inference.evaluation import summarize
p=argparse.ArgumentParser(description=__doc__);p.add_argument('predictions',type=Path);p.add_argument('--options',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
rows=[json.loads(x) for x in args.predictions.read_text().splitlines()]
if any(r['split']!='dev' for r in rows):raise ValueError('Calibrate only on development data')
options=DecodeOptions.load(args.options)
for row in rows:
 ranked=rank_candidates(row.get('candidates',[]),options)
 row['text']=ranked[0]['text'] if ranked else ''
 row['margin']=(ranked[0]['score']-ranked[1]['score'])/max(1,len(ranked[0]['ids'])) if len(ranked)>1 else 0
trials=[]
for threshold in [0,.01,.03,.05,.1,.2,.3,.5]:
 kept=[r for r in rows if r['margin']>=threshold]
 full=[{**r,'text':r['text'] if r['margin']>=threshold else ''} for r in rows]
 trials.append({'threshold':threshold,'coverage':len(kept)/max(1,len(rows)),
                'covered_wer':summarize(kept)['wer'],'full_wer':summarize(full)['wer']})
acceptable=[r for r in trials if r['coverage']>=.8 and r['covered_wer']<=.2]
result={'target':'At least 80% utterance coverage and at most 20% WER among retained utterances',
        'selected_threshold':acceptable[0]['threshold'] if acceptable else None,'trials':trials}
args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
