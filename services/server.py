"""Loopback-only visual caption service with a single bounded inference worker."""
from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from services.inference.model import VisualRecognizer
from services.inference.calibration import DECODERS, select_decoder
from services.inference.stream import FrameBuffer
from services.inference.alignment import WindowCommitter

TOKEN = os.environ.get('RML_TOKEN', '')
DEVICE = os.environ.get('RML_DEVICE', 'auto')
OVERLAP_SECONDS = float(os.environ.get('RML_OVERLAP_SECONDS', '0'))
if OVERLAP_SECONDS not in (0, 0.5, 1):
    raise ValueError('RML_OVERLAP_SECONDS must be 0, 0.5, or 1.')
ORIGINS = {'null', 'file://', 'http://127.0.0.1:5173', 'http://localhost:5173'}
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='visual-model')
state = {'status': 'loading', 'device': None, 'error': None, 'model': 'Auto-AVSR · English'}
recognizer = None
active = False


async def load_model():
    global recognizer
    try:
        recognizer = await asyncio.get_running_loop().run_in_executor(executor, lambda: VisualRecognizer(DEVICE))
        state.update(status='ready', device=recognizer.device, decoding=recognizer.options.asdict(), configuration_note=recognizer.configuration_note)
    except Exception as exc:
        state.update(status='error', error=str(exc))


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(load_model())
    yield
    await task
    if recognizer:
        await asyncio.get_running_loop().run_in_executor(executor, recognizer.close)
    executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=list(ORIGINS), allow_methods=['GET'])


@app.get('/health')
async def health():
    models = [
        {'id': 'auto_avsr', 'label': 'Auto-AVSR · sentence captions',
         'available': state['status'] == 'ready', 'output': 'text'},
    ]
    return state | {'active': active, 'language': 'en', 'visual_only': True, 'models': models}


@app.websocket('/stream')
async def stream(ws: WebSocket):
    global active
    if ws.headers.get('origin') not in ORIGINS:
        await ws.close(code=1008)
        return
    await ws.accept()
    try:
        hello = await asyncio.wait_for(ws.receive_json(), timeout=5)
        if not TOKEN or not secrets.compare_digest(str(hello.get('token', '')), TOKEN):
            await ws.close(code=1008, reason='Invalid local session credential')
            return
        if hello.get('language', 'en') != 'en':
            await ws.close(code=1008, reason='Only English is supported')
            return
        model_id = hello.get('model', 'auto_avsr')
        if model_id != 'auto_avsr':
            await ws.close(code=1008, reason='This model is unavailable until sentence decoding is supported.')
            return
        if active:
            await ws.send_json({'type': 'error', 'message': 'Another recognition session is active.'})
            await ws.close(code=1013)
            return
        calibration_refs = hello.get('calibration')
        if calibration_refs is not None and (
            not isinstance(calibration_refs, dict)
            or any(not isinstance(calibration_refs.get(key), str)
                   or not 30 <= len(calibration_refs[key]) <= 2000
                   for key in ('practice', 'validation'))
        ):
            await ws.close(code=1008, reason='Calibration needs two text passages.')
            return
        if calibration_refs is not None and model_id != 'auto_avsr':
            await ws.close(code=1008, reason='Speaker calibration is available only with Auto-AVSR.')
            return
        if model_id == 'auto_avsr' and state['status'] != 'ready':
            await ws.send_json({'type': 'error', 'message': state.get('error') or 'Auto-AVSR is not ready.'})
            await ws.close(code=1013)
            return
        decoder_profile = hello.get('decoder_profile', 'attention_beam')
        if decoder_profile not in {'hybrid', *DECODERS}:
            await ws.close(code=1008, reason='Invalid decoder profile.')
            return
    except (asyncio.TimeoutError, ValueError, WebSocketDisconnect):
        await ws.close()
        return

    active = True
    session_recognizer = recognizer

    session_recognizer.diagnostics = hello.get('diagnostics') is True
    is_calibrating = calibration_refs is not None
    overlap = 0 if is_calibrating else OVERLAP_SECONDS
    committer = WindowCommitter()
    context_start_segment = 0
    if not is_calibrating:
        session_recognizer.set_decoder_profile(decoder_profile)
    calibration_phase = 'practice' if is_calibrating else None
    calibration_predictions = {
        phase: {decoder: [] for decoder in DECODERS}
        for phase in ('practice', 'validation')
    }
    calibration_finished = False
    calibration_reported = False
    inference_busy = False
    calibration_dropped = False
    epoch = 0
    segment = 0
    buffer = FrameBuffer(seconds=5)
    partial_sent = False
    options = getattr(session_recognizer, 'options', None)
    # Expensive provisional and final passes together can exceed the live compute budget.
    provisional_enabled = options is None or (options.beam_size <= 4 and options.lm_weight == 0)
    queue = asyncio.Queue(maxsize=2 if is_calibrating else 1)
    send_lock = asyncio.Lock()

    async def send(data):
        async with send_lock:
            await ws.send_json(data)

    async def enqueue(final=False, minimum=2, flush=False):
        nonlocal calibration_dropped
        sampled = buffer.sample(minimum)
        if sampled is None:
            return
        frames, start, end = sampled
        job = (epoch, segment, frames, start, end, final, time.monotonic(), calibration_phase, flush)
        if queue.full():
            old = queue.get_nowait()
            queue.task_done()
            if is_calibrating:
                calibration_dropped = True
                await send({'type': 'gap', 'epoch': epoch,
                            'message': 'Calibration inference fell behind; the personalized profile will be rejected.'})
            if old[5] and old[1] != segment:
                await send({'type': 'gap', 'epoch': epoch, 'message': 'Inference fell behind. An interval was skipped.'})
        queue.put_nowait(job)

    async def finish_calibration():
        nonlocal calibration_reported
        if calibration_reported or not is_calibrating or not calibration_finished or inference_busy or not queue.empty():
            return
        calibration_reported = True
        profile, scores = select_decoder(
            calibration_refs['practice'], calibration_refs['validation'],
            calibration_predictions['practice'], calibration_predictions['validation'],
        )
        insufficient_data = calibration_dropped or min(
            len(calibration_predictions[phase]['attention_beam'])
            for phase in ('practice', 'validation')
        ) < 3
        if insufficient_data:
            profile = 'attention_beam'
        session_recognizer.set_decoder_profile(profile)
        await send({'type': 'calibration_complete', 'epoch': epoch,
                    'decoder_profile': profile, 'insufficient_data': insufficient_data, **scores})

    async def infer():
        nonlocal inference_busy
        while True:
            job_epoch, sid, frames, start, end, final, queued, phase, flush = await queue.get()
            try:
                inference_busy = True
                if job_epoch != epoch:
                    inference_busy = False
                    continue
                try:
                    if is_calibrating:
                        result = await asyncio.get_running_loop().run_in_executor(
                            executor, session_recognizer.recognize_candidates, frames)
                        if job_epoch != epoch:
                            inference_busy = False
                            await finish_calibration()
                            continue
                        if phase:
                            for decoder, text in result.items():
                                calibration_predictions[phase][decoder].append(text)
                        await send({'type': 'calibration_progress', 'epoch': job_epoch,
                                    'phase': phase, 'text': result.get('attention_beam', '')})
                        inference_busy = False
                        await finish_calibration()
                        continue
                    result = await asyncio.get_running_loop().run_in_executor(executor, session_recognizer.recognize, frames)
                    if job_epoch != epoch:
                        inference_busy = False
                        continue
                    if overlap and not is_calibrating:
                        if sid < context_start_segment:
                            inference_busy = False
                            continue
                        spans = result.get('word_spans')
                        if result.get('text') and spans is None:
                            result.update(text='', quality='uncertain')
                        elif spans is not None:
                            result['text'] = committer.select(spans, start, end, final,
                                hold_ms=0 if flush else overlap * 500)
                    # Candidate scores are available to the offline evaluator; avoid sending full hypotheses to the UI.
                    result.pop('candidates', None)
                    result.pop('word_spans', None)
                    if result.get('quality') == 'no_speech':
                        print(f"[speech-gate] quality=no_speech ctc_activity={result.get('ctc_activity', 'n/a')}", flush=True)
                    await send({'type': 'caption', 'epoch': job_epoch, 'segment': sid,
                                'start_ms': start, 'end_ms': end, 'final': final,
                                'processing_delay': round(time.monotonic() - queued, 3),
                                'model': model_id, **result})
                    inference_busy = False
                    await finish_calibration()
                except Exception as exc:
                    inference_busy = False
                    if job_epoch == epoch:
                        await send({'type': 'error', 'epoch': epoch, 'message': f'Inference failed: {exc}'})
                    await finish_calibration()
            finally:
                queue.task_done()

    worker = asyncio.create_task(infer())
    try:
        await send({'type': 'ready', 'calibrating': is_calibrating, **state})
        while True:
            message = await asyncio.wait_for(ws.receive(), timeout=90)
            if message['type'] == 'websocket.disconnect':
                break
            if message.get('text') is not None:
                import json
                control = json.loads(message['text'])
                if control.get('type') == 'reset':
                    epoch = int(control['epoch'])
                    segment, buffer, partial_sent = 0, FrameBuffer(seconds=5), False
                    committer = WindowCommitter()
                    context_start_segment = 0
                    while not queue.empty():
                        queue.get_nowait()
                        queue.task_done()
                    if is_calibrating:
                        calibration_dropped = True
                        calibration_predictions = {
                            phase: {decoder: [] for decoder in DECODERS}
                            for phase in ('practice', 'validation')
                        }
                    await send({'type': 'reset', 'epoch': epoch})
                elif control.get('type') == 'finish':
                    await enqueue(final=True, minimum=0.5, flush=True)
                    try:
                        await asyncio.wait_for(queue.join(), timeout=45)
                    except asyncio.TimeoutError:
                        await send({'type':'error', 'message':'Recognition timed out while finishing the last interval.'})
                    else:
                        await send({'type': 'finished', 'epoch': epoch})
                    break
                elif control.get('type') == 'flush':
                    await enqueue(final=True, minimum=0.5, flush=True)
                    buffer, partial_sent = FrameBuffer(seconds=5), False
                    segment += 1
                elif control.get('type') == 'calibration_phase' and is_calibrating:
                    phase = control.get('phase')
                    if phase not in {'practice', 'validation'}:
                        raise ValueError('Invalid calibration passage.')
                    await enqueue(final=True, minimum=2)
                    buffer, partial_sent = FrameBuffer(seconds=5), False
                    calibration_phase = phase
                    await send({'type': 'calibration_phase', 'phase': phase})
                elif control.get('type') == 'finish_calibration' and is_calibrating:
                    await enqueue(final=True, minimum=0.5, flush=True)
                    buffer, partial_sent = FrameBuffer(seconds=5), False
                    calibration_phase = None
                    calibration_finished = True
                    await finish_calibration()
                elif control.get('type') == 'ping':
                    await send({'type': 'pong'})
                continue
            payload = message.get('bytes', b'')
            if not 12 < len(payload) <= 500_000:
                raise ValueError('Invalid frame payload size')
            frame_epoch, timestamp = struct.unpack('<Id', payload[:12])
            if frame_epoch != epoch:
                continue
            frame = cv2.imdecode(np.frombuffer(payload[12:], np.uint8), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[0] > 720 or frame.shape[1] > 960:
                raise ValueError('Invalid frame dimensions')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gap = buffer.push(timestamp, frame)
            if gap:
                if is_calibrating:
                    calibration_dropped = True
                segment += 1
                context_start_segment = segment
                committer = WindowCommitter()
                partial_sent = False
                await send({'type': 'gap', 'epoch': epoch, 'message': 'Capture paused or lost frames; starting a new interval.'})
            duration = (buffer.frames[-1][0] - buffer.frames[0][0]) / 1000
            if duration >= 4:
                await enqueue(final=True)
                if overlap:
                    buffer.retain(overlap)
                    partial_sent = False
                else:
                    buffer, partial_sent = FrameBuffer(seconds=5), False
                segment += 1
            elif duration >= 2 and provisional_enabled and not partial_sent and not inference_busy and queue.empty() and not is_calibrating:
                await enqueue(final=False)
                partial_sent = True
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except Exception as exc:
        with contextlib.suppress(Exception):
            await send({'type': 'error', 'message': str(exc)})
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
            await worker
        active = False
        with contextlib.suppress(Exception):
            await ws.close()


if __name__ == '__main__':
    import uvicorn
    if not TOKEN:
        raise SystemExit('RML_TOKEN is required. Launch via npm run dev or npm start.')
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('RML_PORT', '8765')),
                ws_max_size=500_000, ws_max_queue=4, log_level='warning')
