"""Research adapter for the official visual-only AV-HuBERT checkpoint conversion."""
from pathlib import Path
import sys,time,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]


class AVHubertRecognizer:
    def __init__(self,device='auto',options=None):
        self.beam_size = options.beam_size if options else 4
        import torch
        self.torch=torch
        torch.set_num_threads(4)
        self.device=('mps' if torch.backends.mps.is_available() else 'cpu') if device=='auto' else device
        sys.path.insert(0,str(ROOT/'.cache/avhubert-s2s'))
        from src.model.avhubert2text import AV2TextForConditionalGeneration
        from src.model.av2text_config import AV2TextConfig
        from safetensors.torch import load_file
        path=ROOT/'.cache/models/avhubert-base-hf'
        config=AV2TextConfig.from_pretrained(path,local_files_only=True)
        self.model=AV2TextForConditionalGeneration(config)
        self.model.load_state_dict(load_file(str(path/'model.safetensors')),strict=True)
        self.tokens=json.loads((path/'tokens.json').read_text())
        if self.device=='mps':
            from services.inference.mps import SpatialMaxPool3d
            for module in list(self.model.modules()):
                for name,child in list(module.named_children()):
                    if isinstance(child,torch.nn.MaxPool3d):
                        if child.kernel_size!=(1,3,3) or child.stride!=(1,2,2) or child.padding!=(0,1,1):
                            raise ValueError('Unexpected pooling; do not silently substitute')
                        setattr(module,name,SpatialMaxPool3d())
        self.model.eval().to(self.device)
        self.detectors=[]
        self.diagnostics=False

    def recognize_mouth(self,frames,decoder=None):
        from transformers.modeling_outputs import BaseModelOutput
        started=time.perf_counter();gray=np.stack(frames).astype(np.float32)/255
        if gray.shape[1:]==(96,96):gray=gray[:,4:92,4:92]
        if gray.shape[1:]!=(88,88):raise ValueError('Expected prealigned grayscale mouths')
        video=self.torch.from_numpy((gray-.421)/.165).unsqueeze(0).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            features,_=self.model.model.encoder.extract_finetune({'audio':None,'video':video},mask=False)
            generated=self.model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=features),
                num_beams=self.beam_size,num_return_sequences=self.beam_size,return_dict_in_generate=True,output_scores=True,
                max_new_tokens=min(100,len(frames)),length_penalty=1.0)
        candidates=[]
        for sequence,score in zip(generated.sequences,generated.sequences_scores):
            ids=sequence.cpu().tolist()
            text=''.join(self.tokens[i] for i in ids if i not in (0,1,2,3)).replace('▁',' ').strip()
            candidates.append({'text':text,'av_score':float(score)})
        return {'text':candidates[0]['text'] if candidates else '', 'quality':'experimental','elapsed':round(time.perf_counter()-started,3),
                'device':self.device,'coverage':1.0,'candidates':candidates}

    def recognize(self,frames):
        started=time.perf_counter()
        if not self.detectors:
            import mediapipe as mp
            sys.path.insert(0,str(ROOT/'.cache/auto-avsr'))
            from preparation.detectors.mediapipe.video_process import VideoProcess
            self.cropper=VideoProcess(convert_gray=False)
            self.detectors=[mp.solutions.face_detection.FaceDetection(model_selection=i,min_detection_confidence=.5) for i in (0,1)]
        from services.inference.model import VisualRecognizer
        points,coverage=VisualRecognizer.landmarks(self,frames)
        if coverage<.8:return {'text':'','quality':'no_face','elapsed':time.perf_counter()-started}
        crops=self.cropper(np.stack(frames),points)
        gray=(crops.astype(np.float32) @ np.array([.2989,.5870,.1140])).clip(0,255).astype(np.uint8)
        result=self.recognize_mouth(list(gray));result['elapsed']=round(time.perf_counter()-started,3)
        return result

    def close(self):
        for detector in self.detectors:detector.close()
