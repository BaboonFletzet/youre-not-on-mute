"""Offline final-window simulation matching server overlap ownership."""
from services.inference.alignment import WindowCommitter


def decode_windows(recognizer, frames, prealigned=False, overlap=0, window=4, fps=25):
    size, retained = round(window*fps)+1, round(overlap*fps)
    step = size-retained-1 if retained else size
    if step <= 0: raise ValueError('Overlap must be smaller than window')
    committer = WindowCommitter(); text=[]; elapsed=0; windows=[]
    offset=0
    while offset < len(frames):
        block=frames[offset:offset+size]
        if len(block) < max(2,round(.5*fps)): break
        result=recognizer.recognize_mouth(block) if prealigned else recognizer.recognize(block)
        complete=len(block)==size
        if overlap:
            spans=result.get('word_spans')
            rendered=committer.select(spans or [],offset/fps*1000,(offset+len(block)-1)/fps*1000,
                                     True,overlap*500 if complete else 0)
        else: rendered=result['text']
        text.append(rendered);elapsed+=result['elapsed'];windows.append({'offset':offset,'text':rendered,'elapsed':result['elapsed']})
        if not complete: break
        offset+=step
    return {'text':' '.join(t for t in text if t),'elapsed':elapsed,'quality':'streaming_simulation',
            'windows':windows,'candidates':[], 'note':'Final windows only; live provisional decoding adds compute.'}
