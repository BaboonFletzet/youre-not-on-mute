"""Offline GRID domain-adaptation pilot. Never installs weights into the live app."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import torch
from torch import nn
from services.inference.model import VisualRecognizer,SOURCE
from services.inference.evaluation import summarize
from services.inference.calibration import words
from scripts.evaluate_accuracy import read_frames


class CTCAdapter(nn.Module):
    def __init__(self,base,rank=32):
        super().__init__();self.base=base.requires_grad_(False)
        self.down=nn.Linear(base.in_features,rank,bias=False)
        self.up=nn.Linear(rank,base.out_features,bias=False)
        nn.init.zeros_(self.up.weight)
    def forward(self,x):return self.base(x)+self.up(torch.tanh(self.down(x)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,help='Optional verified train/validation/test JSONL, grouped by recording session')
    p.add_argument('--output',type=Path,default=ROOT/'.cache/accuracy/ctc-adapter')
    p.add_argument('--root',type=Path,default=ROOT/'.cache/datasets/grid/processed')
    p.add_argument('--train-clips',type=int,default=120);p.add_argument('--validation-clips',type=int,default=16)
    p.add_argument('--test-clips',type=int,default=16);p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--device',default='mps');args=p.parse_args()
    torch.manual_seed(29);np.random.seed(29)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    cache=out/'features.pt'
    if cache.exists():
        data=torch.load(cache,weights_only=True,map_location='cpu')
        expected={'train':args.train_clips,'validation':args.validation_clips,'test':args.test_clips,'source':str(args.root.resolve()),'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest() if args.manifest else None}
        legacy = args.manifest is None and args.root.resolve() == (ROOT/'.cache/datasets/grid/processed').resolve() and data['requested'] == {k:expected[k] for k in ('train','validation','test')}
        if data['requested'] != expected and not legacy:raise ValueError('Cache inputs differ; choose a new --output directory before changing inputs')
    else:
        recognizer=VisualRecognizer(args.device)
        import sentencepiece as spm
        processor=spm.SentencePieceProcessor(model_file=str(SOURCE/'spm/unigram/unigram5000.model'))
        mapping={t:i for i,t in enumerate(recognizer.tokens)}
        names=sorted({p.stem for p in (args.root/'s3').glob('*.mpg')},key=lambda x:hashlib.sha256(x.encode()).hexdigest())
        val_names=set(names[:args.validation_clips]);test_pool=sorted({p.stem for p in (args.root/'s4').glob('*.mpg')} - val_names,key=lambda x:hashlib.sha256(x.encode()).hexdigest())
        test_names=set(test_pool[:args.test_clips])
        withheld=val_names|test_names
        selection=[]
        for split,speakers,count in [('train',('s1','s2'),args.train_clips),('validation',('s3',),args.validation_clips),('test',('s4',),args.test_clips)]:
            candidates=[v for speaker in speakers for v in (args.root/speaker).glob('*.mpg')
                        if (v.stem not in withheld if split=='train' else v.stem in (val_names if split=='validation' else test_names))]
            candidates.sort(key=lambda v:hashlib.sha256(str(v.relative_to(args.root)).encode()).hexdigest())
            selection.extend((split,v) for v in candidates[:count])
        custom={}
        if args.manifest:
            records=[json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
            groups={};references={}
            selection=[]
            for record in records:
                split=record['split'];group=record['group'];reference=record['reference'].strip()
                if split not in {'train','validation','test'} or not reference or record.get('label_status')!='verified':
                    raise ValueError('Training requires verified nonempty references and train/validation/test splits')
                if group in groups and groups[group]!=split:raise ValueError('Recording session overlaps splits')
                normalized=' '.join(words(reference))
                if normalized in references and references[normalized]!=split:raise ValueError('A reading passage appears in more than one split')
                groups[group]=split;references[normalized]=split
                video=Path(record['video']).resolve()
                if video in custom:raise ValueError('Duplicate video path in training manifest')
                custom[video]=record;selection.append((split,video))
            if set(groups.values()) != {'train','validation','test'}:raise ValueError('All three splits are required')
        examples=[]
        try:
            for split,video in selection:
                if custom:
                    ref=custom[video]['reference']
                else:
                    align=args.root/'align'/(video.stem+'.align')
                    ref=' '.join(l.split()[2] for l in align.read_text().splitlines() if len(l.split())==3 and l.split()[2] not in {'sil','sp'})
                frames,seconds=read_frames(video)
                recognizer.prealigned=custom.get(video,{}).get('input_kind')=='prealigned_mouth'
                encoded,*_=recognizer._encode(frames,time.perf_counter())
                if encoded is None:
                    print('SKIP',video,flush=True);continue
                labels=[mapping.get(t,1) for t in processor.encode(ref.upper(),out_type=str)]
                examples.append({'id':str(video),'split':split,'reference':ref,
                                 'features':encoded[0].float().cpu(),'labels':labels,'seconds':seconds})
                if len(examples)%20==0:print('CACHED',len(examples),flush=True)
            data={'examples':examples,'base':copy.deepcopy(recognizer.model.ctc.ctc_lo).cpu().state_dict(),
                  'tokens':recognizer.tokens,'requested':{'train':args.train_clips,'validation':args.validation_clips,'test':args.test_clips,'source':str(args.root.resolve()),'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest() if args.manifest else None}}
            torch.save(data,cache)
        finally:recognizer.close()
    examples=data['examples']
    for split in ('train','validation','test'):
        if not any(x['split']==split for x in examples):raise ValueError(f'No {split} clips; refusing an invalid accuracy report')
    dim=examples[0]['features'].shape[-1];vocab=len(data['tokens'])
    base=nn.Linear(dim,vocab);base.load_state_dict(data['base'])
    adapter=CTCAdapter(base).to(args.device)
    optimizer=torch.optim.AdamW([*adapter.down.parameters(),*adapter.up.parameters()],lr=3e-4,weight_decay=.01)
    def evaluate(split,adapted):
        rows=[]
        with torch.inference_mode():
            for ex in examples:
                if ex['split']!=split:continue
                features=ex['features'].to(args.device)
                logits=adapter(features) if adapted else adapter.base(features)
                raw=logits.argmax(-1).cpu().tolist();ids=[];previous=-1
                for token in raw:
                    if token and token!=previous:ids.append(token)
                    previous=token
                text=''.join(data['tokens'][i] for i in ids if i!=vocab-1).replace('▁',' ').strip()
                rows.append({'reference':ex['reference'],'text':text,'elapsed':0,'video_seconds':ex['seconds']})
        metrics=summarize(rows)
        metrics['decode_p95_seconds']=None
        metrics['real_time_factor']=None
        metrics['timing_note']='Cached encoder features; this is not an end-to-end timing benchmark'
        return metrics
    baseline=evaluate('validation',False);best=baseline['wer'];chosen=None;history=[]
    training=[x for x in examples if x['split']=='train']
    if not training:raise ValueError('No training clips')
    for epoch in range(args.epochs):
        for index in torch.randperm(len(training)).tolist():
            ex=training[index];features=ex['features'].to(args.device)
            logits=adapter(features)
            logp=logits.log_softmax(-1)
            # CPU CTCLoss is differentiable back to the MPS adapter.
            loss=nn.functional.ctc_loss(logp.cpu().unsqueeze(1),torch.tensor(ex['labels']),
                torch.tensor([len(features)]),torch.tensor([len(ex['labels'])]),reduction='mean')
            with torch.no_grad():teacher=adapter.base(features).softmax(-1)
            regularization=nn.functional.kl_div(logp,teacher,reduction='batchmean')
            total=loss+.5*regularization
            if not torch.isfinite(total):raise ValueError('Nonfinite adapter loss')
            optimizer.zero_grad();total.backward();nn.utils.clip_grad_norm_(adapter.parameters(),1);optimizer.step()
        score=evaluate('validation',True);history.append(score);print('EPOCH',epoch+1,json.dumps(score),flush=True)
        if score['wer'] < best:
            best=score['wer'];chosen={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
    report={'domain':'User recordings' if args.manifest else 'GRID fixed six-word grammar; not evidence of conversational improvement',
            'split_policy':'disjoint sessions and passages' if args.manifest else 'train s1/s2, validation s3, test s4; validation/test utterance filenames excluded from training',
            'validation_baseline':baseline,'history':history,'adapter_selected':chosen is not None,
            'test_baseline':evaluate('test',False)}
    if chosen:
        adapter.load_state_dict(chosen);report['test_adapted']=evaluate('test',True)
        torch.save({'state_dict':chosen,'tokens':data['tokens'],'domain':'GRID research only'},out/'adapter.pth')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
