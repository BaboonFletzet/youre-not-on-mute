"""Check whether the released visual architecture can emit sentence-length phonemes."""
import argparse, contextlib, io, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='auto');args=p.parse_args()
from services.inference.vallr import VALLRRecognizer
recognizer=VALLRRecognizer(args.device)
try:
    with recognizer.torch.inference_mode(),contextlib.redirect_stdout(io.StringIO()):
        logits,_=recognizer.model(recognizer.torch.zeros((1,16,3,224,224),device=recognizer.device))
    report={'input_frames':16,'output_shape':list(logits.shape),'maximum_collapsed_phonemes':logits.shape[1],
            'sentence_decoder_viable':False if logits.shape[1] <= 2 else None,
            'note':'Shape audit only, not an accuracy test. Current fixed 16-frame visual path cannot represent more phonemes than output timesteps.'}
    path=ROOT/'.cache/accuracy/vallr-audit.json';path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
finally:recognizer.close()
