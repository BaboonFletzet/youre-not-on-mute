import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { ArrowDownToLine, ArrowUpRight, Check, ChevronDown, Cpu, FileVideo, Focus, Globe2, LoaderCircle, Monitor, Play, Plus, ScanFace, ShieldCheck, Square, Video, X } from 'lucide-react';
import './style.css';

type Source = { id: string; name: string; thumbnail: string };
type Config = { token: string; baseUrl: string; streamUrl: string };
type Health = { status: string; device?: string; error?: string; model?: string; models?: { id: string; label: string; available: boolean; output: string }[] };
type Caption = { input_quality?: { small_face?: boolean; inter_eye_pixels?: number }; mouth_preview?: string; score_margin?: number; segment: number; text: string; final: boolean; start_ms: number; end_ms: number; elapsed: number; processing_delay: number; quality: string; coverage: number };
type Crop = { x: number; y: number; w: number; h: number };
type CalibrationTexts = { practice: string; validation: string };
type CalibrationReport = { decoder_profile: string; insufficient_data: boolean; validation_wer: Record<string, number> };
declare global { interface Window { desktop?: { config(): Promise<Config>; sources(): Promise<Source[]>; selectSource(id: string): Promise<boolean>; sample(): Promise<Uint8Array>; captureSettings(): Promise<void> } } }

const fullCrop = { x: 0, y: 0, w: 1, h: 1 };
const calibrationTexts: CalibrationTexts = {
  practice: 'On a quiet morning, a small boat moved slowly across the lake. The water reflected the pale sky, while a few birds crossed above the trees. Along the shore, people opened their windows and began the day. A bell rang in the distance, and the town grew busy one street at a time. Nothing happened quickly, but every small sound and movement made the morning feel alive.',
  validation: 'After lunch, the weather changed and dark clouds gathered over the hills. A warm wind pushed leaves along the road. Neighbors carried their bags inside and waited by the windows. Soon the rain passed, sunlight returned, and the wet pavement shone. The evening was calm again, with long shadows stretching across the garden and a bright star appearing above the rooftops.'
};
const timeLabel = (ms: number) => { const s = Math.max(0, Math.floor(ms / 1000)); return `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`; };

function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [health, setHealth] = useState<Health>({ status: 'connecting' });
  const [sources, setSources] = useState<Source[] | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [connectingSource, setConnectingSource] = useState(false);
  const [captureDenied, setCaptureDenied] = useState(false);
  const [sourceName, setSourceName] = useState('No video selected');
  const [hasVideo, setHasVideo] = useState(false);
  const [videoReady, setVideoReady] = useState(false);
  const [isFile, setIsFile] = useState(false);
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [finishing, setFinishing] = useState(false);
  const [captions, setCaptions] = useState<Caption[]>([]);
  const [notice, setNotice] = useState('');
  const [showModelInput, setShowModelInput] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [crop, setCrop] = useState<Crop>(fullCrop);
  const [selecting, setSelecting] = useState(false);
  const [help, setHelp] = useState(false);
  const [connected, setConnected] = useState(false);
  const [calibrationSetup, setCalibrationSetup] = useState(false);
  const [calibrationPhase, setCalibrationPhase] = useState<'practice' | 'validation' | 'scoring' | 'done' | null>(null);
  const [calibrationReport, setCalibrationReport] = useState<CalibrationReport | null>(null);
  const [decoderProfile, setDecoderProfile] = useState('attention_beam');
  const modelSelection = 'auto_avsr';
  const calibrationRef = useRef<CalibrationTexts | null>(null);
  const video = useRef<HTMLVideoElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const preview = useRef<HTMLDivElement>(null);
  const socket = useRef<WebSocket | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const fileUrl = useRef<string | null>(null);
  const epoch = useRef(0);
  const sending = useRef(false);
  const startedAt = useRef(0);
  const videoFrameHandle = useRef<number | null>(null);
  const cropRef = useRef(crop);
  const runningRef = useRef(false);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const sessionStart = useRef<number | null>(null);
  cropRef.current = crop;

  useEffect(() => {
    let closed = false;
    window.desktop?.config().then(c => { if (!closed) setConfig(c); });
    return () => { closed = true; stopCapture(); };
  }, []);

  useEffect(() => {
    if (!config) return;
    let closed = false;
    const poll = async () => {
      try { const r = await fetch(`${config.baseUrl}/health`); const h = await r.json(); if (!closed) setHealth(h); }
      catch { if (!closed) setHealth({ status: 'connecting' }); }
    };
    void poll(); const timer = setInterval(poll, 2000);
    return () => { closed = true; clearInterval(timer); };
  }, [config]);

  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => setElapsed(Math.floor((performance.now() - startedAt.current) / 1000)), 500);
    return () => clearInterval(timer);
  }, [running]);

  function stopReading() {
    runningRef.current = false;
    setRunning(false); setStarting(false); setFinishing(false); setConnected(false);
    if (videoFrameHandle.current !== null) video.current?.cancelVideoFrameCallback(videoFrameHandle.current);
    videoFrameHandle.current = null;
    socket.current?.close(); socket.current = null;
  }

  function finishReading() {
    if (socket.current?.readyState !== WebSocket.OPEN) { stopReading(); return; }
    runningRef.current = false;
    if (videoFrameHandle.current !== null) video.current?.cancelVideoFrameCallback(videoFrameHandle.current);
    videoFrameHandle.current = null;
    setFinishing(true); setNotice('Finishing the last interval…');
    if (calibrationRef.current) { calibrationRef.current = null; setCalibrationPhase(null); }
    socket.current.send(JSON.stringify({ type: 'finish' }));
  }

  function stopCapture() {
    stopReading();
    stream.current?.getTracks().forEach(t => t.stop()); stream.current = null;
    if (fileUrl.current) URL.revokeObjectURL(fileUrl.current);
    fileUrl.current = null;
  }

  async function pickSource() {
    if (!window.desktop) { setNotice('Open the desktop app with npm run dev to capture another window.'); return; }
    setSourceLoading(true); setNotice('');
    try { setSources(await window.desktop.sources()); setCaptureDenied(false); }
    catch { setCaptureDenied(true); setNotice('macOS is denying screen capture. Enable Electron in Screen & System Audio Recording settings. Because this app was launched with npm, also enable the terminal you use (iTerm or Terminal), then quit and reopen both apps. Node itself is only the launcher. If access is still denied, this ad hoc signed development build may need an Apple Development signature.'); }
    finally { setSourceLoading(false); }
  }

  async function chooseSource(source: Source) {
    if (connectingSource) return;
    setConnectingSource(true);
    try {
      stopCapture();
      await window.desktop!.selectSource(source.id);
      const media = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: { ideal: 25, max: 30 } }, audio: false });
      stream.current = media; setSources(null); setIsFile(false); setHasVideo(true); setVideoReady(false);
      setSourceName(source.name); setCrop(fullCrop); setCaptions([]); setNotice('');
      setDecoderProfile('attention_beam');
      setCalibrationPhase(null); setCalibrationReport(null);
      const player = video.current!; player.removeAttribute('src'); player.srcObject = media;
      media.getVideoTracks()[0].addEventListener('ended', () => { stopReading(); setHasVideo(false); setNotice('Screen sharing ended. Select a source to continue.'); });
      await player.play();
    } catch (e) {
      setSources(null);
      const reason = e instanceof Error ? e.message : String(e);
      setNotice(`Could not start capture of “${source.name}”: ${reason}`);
    } finally { setConnectingSource(false); }
  }

  async function chooseFile(file?: File) {
    if (!file) return;
    stopCapture(); setSources(null); setCaptureDenied(false); setIsFile(true); setHasVideo(true); setVideoReady(false); setCrop(fullCrop);
    setSourceName(file.name); setCaptions([]); setNotice('');
    setDecoderProfile('attention_beam');
    setCalibrationPhase(null); setCalibrationReport(null);
    fileUrl.current = URL.createObjectURL(file);
    const player = video.current!; player.srcObject = null; player.src = fileUrl.current;
    player.load();
  }

  async function trySample() {
    try {
      const bytes = await window.desktop!.sample();
      await chooseFile(new File([new Uint8Array(bytes)], 'English research sample.mp4', { type: 'video/mp4' }));
    } catch { setNotice('The optional sample is not installed. Open your own video file, or see readme.MD for the sample download.'); }
  }

  function resetContext() {
    epoch.current += 1; sessionStart.current = null;
    if (socket.current?.readyState === WebSocket.OPEN) socket.current.send(JSON.stringify({ type: 'reset', epoch: epoch.current }));
    setCaptions([]); setNotice('Context reset. Gathering new video…');
  }

  async function startReading(calibration?: CalibrationTexts) {
    if (!config || !video.current || !selectedModel?.available) return;
    if (!calibration) { setCalibrationPhase(null); setCalibrationReport(null); }
    stopReading(); setStarting(true); setNotice('Gathering the first few seconds of video…'); setCaptions([]);
    epoch.current = 0; sessionStart.current = null; sending.current = false;
    startedAt.current = performance.now(); setElapsed(0); setFrameCount(0);
    const ws = new WebSocket(config.streamUrl); socket.current = ws;
    calibrationRef.current = calibration || null;
    ws.onopen = () => ws.send(JSON.stringify({ token: config.token, language: 'en', model: modelSelection, diagnostics: showModelInput, decoder_profile: decoderProfile, ...(calibration ? { calibration } : {}) }));
    ws.onerror = () => setNotice('Could not connect to the local recognition service.');
    ws.onclose = () => { if (socket.current === ws) stopReading(); };
    ws.onmessage = async event => {
      const message = JSON.parse(event.data);
      if (message.type === 'ready') {
        runningRef.current = true; setStarting(false); setRunning(true); setConnected(true);
        if (calibrationRef.current) setCalibrationPhase('practice');
        if (isFile && video.current) {
          try { video.current.currentTime = 0; await video.current.play(); }
          catch { stopReading(); setNotice('This video could not be played. Try an H.264 MP4 file.'); return; }
        }
        sendFrames();
      } else if (message.type === 'finished') {
        stopReading();
        setNotice(current => current === 'Finishing the last interval…' ? 'Session finished.' : current);
      } else if (message.type === 'caption' && message.epoch === epoch.current) {
        setCaptions(items => [...items.filter(c => c.segment !== message.segment), message].sort((a, b) => a.segment - b.segment).slice(-100));
        setNotice(message.input_quality?.small_face ? 'The face is small in the source video. Enlarge the speaker or use a higher-resolution video for clearer lips.' : message.quality === 'uncertain' ? 'This interval is unclear. Try a larger, front-facing view of the mouth.' : message.quality === 'no_face' ? 'Keep one face clearly visible. Use Select region to isolate the speaker.' : message.quality === 'no_speech' ? 'The visual speech detector found too little speech-like activity in this interval.' : '');
      } else if (message.type === 'calibration_phase') {
        setCalibrationPhase(message.phase);
      } else if (message.type === 'calibration_progress') {
        setNotice(`${message.phase === 'practice' ? 'Practice' : 'Validation'} passage · ${message.text || 'interval processed'}`);
      } else if (message.type === 'calibration_complete') {
        setCalibrationReport(message); setDecoderProfile(message.decoder_profile); setCalibrationPhase('done');
        stopReading(); setNotice(message.insufficient_data ? 'Calibration needs more clear speech; keeping the standard decoder.' : `Calibration complete. ${message.decoder_profile} decoding selected.`);
      } else if (message.type === 'error' || message.type === 'gap') setNotice(message.message);
    };
  }

  function beginCalibration() {
    calibrationRef.current = calibrationTexts;
    setCalibrationReport(null); setCalibrationSetup(false); setCalibrationPhase(null);
    void startReading(calibrationTexts);
  }

  function nextCalibrationPassage() {
    if (socket.current?.readyState === WebSocket.OPEN) {
      socket.current.send(JSON.stringify({ type: 'calibration_phase', phase: 'validation' }));
      setCalibrationPhase('validation');
      setNotice('Read the validation passage at a natural pace.');
    }
  }

  function finishCalibration() {
    setCalibrationPhase('scoring'); setNotice('Scoring the held-out passage…');
    if (socket.current?.readyState === WebSocket.OPEN) socket.current.send(JSON.stringify({ type: 'finish_calibration' }));
  }

  function sendFrames() {
    const player = video.current!;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { alpha: false })!;
    const schedule = () => { videoFrameHandle.current = player.requestVideoFrameCallback(tick); };
    let nextSample = -Infinity;
    let samplingEpoch = epoch.current;
    const tick = (_now: number, metadata: VideoFrameCallbackMetadata) => {
      if (!runningRef.current) return;
      schedule();
      const ws = socket.current;
      if (!ws || ws.readyState !== WebSocket.OPEN || sending.current || player.readyState < 2 || player.paused || player.ended) return;
      const mediaMs = metadata.mediaTime * 1000;
      if (samplingEpoch !== epoch.current) { samplingEpoch = epoch.current; nextSample = -Infinity; }
      if (mediaMs + 0.5 < nextSample || ws.bufferedAmount > 1_000_000) return;
      nextSample = nextSample === -Infinity ? mediaMs + 40 : Math.max(nextSample + 40, mediaMs);
      const c = cropRef.current;
      const sw = player.videoWidth * c.w, sh = player.videoHeight * c.h;
      if (sw < 32 || sh < 32) return;
      // Preserve more face/lip detail; the local service accepts up to 960×720.
      const scale = Math.min(1, 960 / sw, 720 / sh);
      canvas.width = Math.round(sw * scale); canvas.height = Math.round(sh * scale);
      ctx.drawImage(player, c.x * player.videoWidth, c.y * player.videoHeight, sw, sh, 0, 0, canvas.width, canvas.height);
      const frameEpoch = epoch.current;
      sessionStart.current ??= mediaMs;
      const timestamp = mediaMs - sessionStart.current;
      sending.current = true;
      canvas.toBlob(async blob => {
        try {
          if (!blob || frameEpoch !== epoch.current || !runningRef.current || socket.current !== ws || ws.readyState !== WebSocket.OPEN) return;
          const data = await blob.arrayBuffer();
          if (frameEpoch !== epoch.current || !runningRef.current || socket.current !== ws || ws.readyState !== WebSocket.OPEN) return;
          const packet = new Uint8Array(12 + data.byteLength);
          const view = new DataView(packet.buffer);
          view.setUint32(0, frameEpoch, true); view.setFloat64(4, timestamp, true); packet.set(new Uint8Array(data), 12);
          ws.send(packet); setFrameCount(n => n + 1);
        } finally { if (socket.current === ws) sending.current = false; }
      }, 'image/jpeg', 0.92);
    };
    schedule();
  }

  function point(event: React.PointerEvent) {
    const box = preview.current!.getBoundingClientRect();
    return { x: Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)), y: Math.max(0, Math.min(1, (event.clientY - box.top) / box.height)) };
  }

  function exportTranscript() {
    const text = captions.filter(c => c.text).map(c => `[${timeLabel(c.start_ms)}–${timeLabel(c.end_ms)}] ${c.text}${c.final ? '' : ' [provisional]'}`).join('\n');
    const heading = "You're NOT on Mute! — English visual-only transcript";
    const url = URL.createObjectURL(new Blob([`${heading}\n${sourceName}\n\n${text}\n`], { type: 'text/plain' }));
    const a = document.createElement('a'); a.href = url; a.download = 'youre-not-on-mute-transcript.txt'; a.click(); URL.revokeObjectURL(url);
  }

  const latest = captions.filter(c => c.text).at(-1);
  const inferenceTime = captions.at(-1)?.elapsed;
  const selectedModel = health.models?.find(option => option.id === modelSelection);
  const ready = selectedModel?.available ?? health.status === 'ready';
  return <div className="app-shell">
    <div className="titlebar"><span className="titlebar-name">YOU'RE NOT ON MUTE!</span><span className="titlebar-version">PREVIEW <span>0.2</span></span></div>
    <header>
      <a className="brand" href="#" aria-label="You're NOT on Mute! home"><span className="brand-icon"><ScanFace size={23} strokeWidth={1.6}/></span><span className="brand-wordmark">You're <strong>NOT</strong> on Mute!</span></a>
      <nav><span className="nav-active">Workspace</span><button onClick={() => setHelp(true)}>How it works <ArrowUpRight size={13}/></button></nav>
      <div className="local-badge"><span className="pulse-dot"/> Local processing</div>
    </header>
    <main>
      <section className="page-heading"><div><h1>Read visible speech.</h1><p>Select a video source and focus on one speaker to generate English captions.</p></div></section>
      <div className="workspace">
        <section className="video-card">
          <div className="card-heading"><span><span className={running ? 'status-light live' : 'status-light'}/>{running ? 'READING LIPS' : 'VIDEO SOURCE'}</span><span className="source-title" title={sourceName}>{sourceName}</span><button className="small-button" onClick={pickSource} disabled={sourceLoading}><Plus size={14}/>{sourceLoading ? 'Finding windows…' : 'Choose source'}</button></div>
          <div className={`video-stage ${hasVideo ? 'has-video' : ''}`}>
            <div className="video-content" ref={preview} style={hasVideo && video.current?.videoWidth ? { '--video-ratio': video.current.videoWidth / video.current.videoHeight } as React.CSSProperties : undefined}
              onPointerDown={e => { if (!selecting) return; e.currentTarget.setPointerCapture(e.pointerId); drag.current = point(e); setCrop({ ...drag.current, w: 0, h: 0 }); }}
              onPointerMove={e => { if (!drag.current) return; const p = point(e), s = drag.current; setCrop({ x: Math.min(p.x, s.x), y: Math.min(p.y, s.y), w: Math.abs(p.x - s.x), h: Math.abs(p.y - s.y) }); }}
              onPointerUp={() => { if (!drag.current) return; drag.current = null; setSelecting(false); if (cropRef.current.w < 0.08 || cropRef.current.h < 0.08) setCrop(fullCrop); if (runningRef.current) resetContext(); }}>
              <video ref={video} muted playsInline controls={isFile && !selecting} className={hasVideo ? '' : 'hidden'}
                onLoadedData={() => setVideoReady(true)} onError={() => { stopReading(); setVideoReady(false); setNotice('This video format is not supported by the player. Use H.264 MP4; conversion instructions are in readme.MD.'); }} onSeeked={() => { if (runningRef.current) resetContext(); }}
                onEnded={() => { if (socket.current?.readyState === WebSocket.OPEN) { socket.current.send(JSON.stringify({ type: 'flush' })); setNotice('Video ended. Finishing the last interval…'); } }}/>
              {hasVideo && (selecting || crop.w !== 1 || crop.h !== 1) && <div className={`crop-layer ${selecting ? 'selecting' : ''}`}><div className="crop-box" style={{ left: `${crop.x * 100}%`, top: `${crop.y * 100}%`, width: `${crop.w * 100}%`, height: `${crop.h * 100}%` }}><span>SELECTED REGION</span></div></div>}
            </div>
            {!hasVideo && <div className="empty-video"><div className="scan-illustration"><Monitor size={32} strokeWidth={1.4}/></div><h2>Choose a video source</h2><p>Choose a window with a playing video,<br/>or open a local video file.</p><div className="source-actions"><button className="primary-button" onClick={pickSource}><Monitor size={16}/> Choose a window</button><button className="text-button" onClick={() => fileInput.current?.click()}>Open a video file <ArrowUpRight size={14}/></button></div><button className="sample-link" onClick={trySample}>Try a sample clip →</button></div>}
            {running && latest && <div className="caption-overlay"><span>{latest.text}</span></div>}
            {selecting && <div className="selection-hint">Crop tightly around one face, from forehead to chin. Keep the mouth clear.</div>}
          </div>
          <div className="video-toolbar"><div><button className={`tool-button ${selecting ? 'selected' : ''}`} disabled={!hasVideo} onClick={() => setSelecting(s => !s)}><Focus size={16}/>Select region</button><button className="tool-button" disabled={!hasVideo} onClick={() => { setCrop(fullCrop); if (runningRef.current) resetContext(); }}>Reset</button><button className="tool-button file-tool" onClick={() => fileInput.current?.click()}><FileVideo size={16}/>Open file</button></div><span className="audio-off"><Video size={14}/> VIDEO ONLY</span></div>
          <div className="input-diagnostics"><label><input type="checkbox" checked={showModelInput} disabled={running} onChange={e => setShowModelInput(e.target.checked)}/> Show what the model sees</label>{showModelInput && captions.at(-1)?.mouth_preview && <figure><img src={captions.at(-1)!.mouth_preview} alt="Eight grayscale mouth frames from the model input"/><figcaption>Actual mouth crops · sampled through the last interval · normalized before inference</figcaption></figure>}</div>
          <div className="session-controls"><div><span className={`session-indicator ${running ? 'on' : ''}`}/><span>{running ? calibrationPhase ? 'Speaker calibration' : 'Session in progress' : videoReady ? 'Ready' : 'Select a source to begin'}</span><span className="session-time">{timeLabel(elapsed * 1000)}</span></div><div className="session-actions"><span className="model-picker"><span>MODEL</span><b>Auto-AVSR · sentence captions</b></span>{!running && <button className="tool-button" disabled={!hasVideo || !videoReady || !ready} onClick={() => setCalibrationSetup(true)}>Calibrate speaker · 1 min</button>}<button className={running ? 'stop-button' : 'primary-button'} disabled={starting || finishing || (!running && (!hasVideo || !videoReady || !ready))} onClick={() => running ? finishReading() : void startReading()}>{starting || finishing ? <><LoaderCircle size={14} className="spin"/> {finishing ? 'Finishing…' : 'Starting…'}</> : running ? <><Square size={13} fill="currentColor"/> Stop reading</> : <><Play size={14} fill="currentColor"/> Start lip reading</>}</button></div></div>
        </section>
        <aside className="transcript-card"><div className="card-heading"><span>{calibrationPhase ? 'SPEAKER CALIBRATION' : 'LIVE TRANSCRIPT'}</span><span className="language"><Globe2 size={13}/> English <ChevronDown size={12}/></span></div><div className="transcript-body">
          {calibrationPhase ? <div className="calibration-copy">{calibrationPhase === 'practice' ? <><span className="calibration-label">PRACTICE PASSAGE</span><p>{calibrationTexts.practice}</p><button className="primary-button" onClick={nextCalibrationPassage}>Next: validation passage <ArrowUpRight size={13}/></button></> : calibrationPhase === 'validation' ? <><span className="calibration-label">HELD-OUT VALIDATION</span><p>{calibrationTexts.validation}</p><button className="primary-button" onClick={finishCalibration}>Finish and score <Check size={13}/></button></> : calibrationPhase === 'scoring' ? <><LoaderCircle size={18} className="spin"/><h3>Scoring the held-out passage…</h3><p>This takes a few seconds.</p></> : <><span className="calibration-label">CALIBRATION RESULT</span><h3>{calibrationReport?.insufficient_data ? 'Not enough clear speech' : `${calibrationReport?.decoder_profile} decoder selected`}</h3><p>{calibrationReport?.insufficient_data ? 'Keeping the standard decoder. Try again with a larger, front-facing face and read each passage for longer.' : 'The selected profile passed the held-out check. Model weights were not changed.'}</p><div className="calibration-scores">{Object.keys(calibrationReport?.validation_wer || {}).map(key => <span key={key}>{key}<b>{Math.round((calibrationReport?.validation_wer?.[key] || 0) * 100)}% held-out WER</b></span>)}</div><button className="tool-button" onClick={() => { setCalibrationPhase(null); setCalibrationReport(null); calibrationRef.current = null; }}>Done</button></>}</div> : captions.some(c => c.text) ? <div className="caption-list">{captions.filter(c => c.text).map(c => <article key={c.segment} className={c.final ? '' : 'provisional'}><div><time>{timeLabel(c.start_ms)}</time><span>{c.final ? <><Check size={10}/> COMPLETE</> : 'PROVISIONAL'}</span></div><p dir="auto">{c.text}</p></article>)}</div> : <div className="empty-transcript"><div className="text-lines"><i/><i/><i/></div><h3>No captions yet</h3><p>Start lip reading to see captions here. Processing takes a few seconds.</p><span>Experimental captions may contain errors.</span></div>}
          {notice && <div className="notice" role="status">{running && !latest && <LoaderCircle size={13} className="spin"/>}<div>{notice}{captureDenied && <button className="settings-link" onClick={() => window.desktop?.captureSettings()}>Open Screen &amp; System Audio Recording settings <ArrowUpRight size={12}/></button>}</div></div>}
        </div><div className="transcript-footer"><span>{captions.filter(c => c.text).length} intervals</span><button className="tool-button" disabled={!captions.some(c => c.text)} onClick={exportTranscript}><ArrowDownToLine size={14}/> Export .txt</button></div></aside>
      </div>
      <div className="bottom-grid"><section className="model-card"><div className="model-icon"><Cpu size={22}/></div><div><div className="model-title">English visual speech model <span className={`model-state ${ready ? 'ready' : ''}`}>{ready ? 'READY' : selectedModel ? 'UNAVAILABLE' : 'LOADING'}</span></div><p>{health.error || (ready ? `English captions · ${calibrationReport && !calibrationReport.insufficient_data ? 'Speaker profile active' : 'Local recognition'} · ${health.device === 'mps' ? 'Apple Metal GPU' : 'CPU'} · Still experimental` : 'Preparing the local recognition engine…')}</p></div><div className="model-metrics"><span>{inferenceTime ? `${inferenceTime.toFixed(1)}s` : '—'}<small>last decode</small></span><span>{running && elapsed > 0 ? Math.round(frameCount / elapsed) : '—'}<small>frames / sec</small></span></div></section><section className="tip-card"><ScanFace size={22}/><div><h3>Keep the speaker clearly visible.</h3><p>Crop tightly, face the camera, and keep the mouth unobstructed.</p></div></section></div>
      <footer><span><ShieldCheck size={14}/> Your video stays on this device. No audio is captured.</span><span>EARLY PREVIEW <i/> {connected ? 'LOCAL SESSION CONNECTED' : 'ENGLISH · LOCAL PROCESSING'}</span></footer>
    </main>
    <input type="file" accept="video/*" ref={fileInput} className="hidden" onChange={e => { void chooseFile(e.target.files?.[0]); e.target.value = ''; }}/>
    {calibrationSetup && <div className="modal-backdrop" onClick={() => setCalibrationSetup(false)}><section className="help-modal calibration-setup" onClick={e => e.stopPropagation()}><div className="modal-heading"><div><div className="eyebrow">ABOUT ONE MINUTE · SAME SPEAKER</div><h2>Read two short passages.</h2><p>Use a selected video source that shows the person’s face while they read. Keep the speaker and camera position the same as the session you want to improve. Read each passage naturally; the practice passage tunes the decoder choice, and the second passage is held out to reject overfitting.</p></div><button className="icon-button" aria-label="Close calibration" onClick={() => setCalibrationSetup(false)}><X size={20}/></button></div><div className="calibration-passages"><article><span className="calibration-label">PRACTICE</span><p>{calibrationTexts.practice}</p></article><article><span className="calibration-label">VALIDATION</span><p>{calibrationTexts.validation}</p></article></div><p className="help-footnote">Keep the face large in frame, facing the camera, with the mouth unobstructed. Calibration compares three existing decoders; it does not update the neural-network weights. It needs at least three usable intervals from each passage.</p><button className="primary-button" disabled={!ready || !videoReady} onClick={beginCalibration}><Play size={14} fill="currentColor"/> Start calibration</button></section></div>}
    {sources && <div className="modal-backdrop" onClick={() => { if (!connectingSource) setSources(null); }}><section className="source-modal" onClick={e => e.stopPropagation()}><div className="modal-heading"><div><div className="eyebrow">VIDEO SOURCE</div><h2>{connectingSource ? 'Starting capture…' : 'Select a source'}</h2><p>{connectingSource ? 'Connecting to the selected window or display.' : 'Select a window or display. You can crop it next.'}</p></div><button className="icon-button" aria-label="Close source picker" disabled={connectingSource} onClick={() => setSources(null)}><X size={20}/></button></div><div className="source-grid">{sources.map(s => <button key={s.id} disabled={connectingSource} onClick={() => { void chooseSource(s); }}><img src={s.thumbnail} alt=""/><span><Monitor size={14}/>{s.name}</span></button>)}</div>{!sources.length && <p>No sources are available. Enable Electron and the terminal used to launch it in macOS Screen & System Audio Recording settings, then restart both apps.</p>}</section></div>}
    {help && <div className="modal-backdrop" onClick={() => setHelp(false)}><section className="help-modal" onClick={e => e.stopPropagation()}><div className="modal-heading"><div><div className="eyebrow">HELP</div><h2>Using the app</h2></div><button className="icon-button" aria-label="Close help" onClick={() => setHelp(false)}><X size={20}/></button></div><ol><li><b>Choose your video.</b><p>Capture a browser or player window, or open a local file.</p></li><li><b>Find one clear speaker.</b><p>Use Select region to crop tightly around one face, from forehead to chin, with the mouth unobstructed.</p></li><li><b>Start lip reading.</b><p>The English model reads short video intervals. Captions arrive after a few seconds and can be inaccurate.</p></li></ol><p className="help-footnote">Everything runs locally. This early preview does not identify speakers, listen to audio, translate speech, or guarantee accurate transcription.</p></section></div>}
  </div>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
