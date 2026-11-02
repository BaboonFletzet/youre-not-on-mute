# You're Not on Mute

A local desktop prototype that reads visible English speech from video. Select a browser/player window or open a video file and isolate one speaker. Auto-AVSR produces English sentence captions. Audio is never passed to the model. Processing runs on this Mac.

The architecture and longer-term evaluation plan are in [docs/v0.md](docs/v0.md). This first implementation uses short intervals to establish working local capture and inference; it remains an early preview. On 160 public evaluation clips, the new decoder reduced word error rate from 28.3% to 26.3%, with fewer insertions and more omissions. General live reliability is not established. Measurements and limitations are in [docs/accuracy-report.md](docs/accuracy-report.md).

## This machine

- Apple M1 Pro, 16 GB unified memory, macOS 15.6.1 / ARM64.
- Python 3.12.3 and PyTorch 2.7.1. Metal/MPS is the preferred inference device.
- Node.js 24.11.0 and npm 11.6.1.
- FFmpeg/ffprobe from Homebrew, for fixture inspection and conversion. The app reads file playback through Chromium; the benchmark reads video frames with PyAV.
- Git, to fetch the pinned reference implementation.

Python dependencies, including the installed transitive packages and test tooling, are pinned in **`requirements.txt`**. JavaScript dependencies are declared in **`package.json`** and resolved in **`package-lock.json`**. No global Python packages are required. Electron embeds Chromium; it can capture other browsers or native media players without a browser extension.

## Install

From the project directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
npm ci
.venv/bin/python scripts/setup_model.py
.venv/bin/python scripts/setup_accuracy_assets.py lm
```

The visual model setup downloads about 956 MiB; local candidate rescoring adds about 523 MiB of GPT-2 assets. The visual setup clones the authors' code into `.cache/`. It requires network access during setup. After setup, video processing uses a local service on `127.0.0.1`; media is not uploaded to an external server.

The VALLR visual checkpoint is retained for research, but is not selectable in the app because its downloaded checkpoint only produces phonemes and no compatible phoneme-to-text decoder weights are available. Its source includes a training script for an LLM adapter, but not a ready-to-use adapter checkpoint. The setup command is:

```sh
.venv/bin/python scripts/setup_vallr.py
```

This downloads its 696 MiB checkpoint and pinned source into `.cache/`, verifying the checkpoint SHA-256. VALLR's code and checkpoint are CC BY-NC 4.0.

## Run

```sh
npm run dev
```

This starts Vite and Electron. Electron starts the native Python inference service on an available loopback port, supplies a random session credential, and stops the service when the app quits. Allow several seconds for the model to load; the model card changes to **READY**.

For a build without Vite:

```sh
npm run build
npm start
```

1. Choose a window, or open a local video file. The project directory is `~/Projects/youre-not-on-mute`; the app, window title, logo, and transcript exports now use **You're Not on Mute**.
2. Use **Select region** to crop tightly around one person's face from forehead to chin, keeping the mouth unobstructed. This gives the face detector more pixels to work with.
3. Click **Start lip reading**. File playback starts from the beginning, muted.
4. Read the transcript and captions. **Stop reading** finishes the last usable interval before closing the session. Export the transcript as plain text when ready. Enable **Show what the model sees** before starting to inspect the actual mouth crops.

**Calibrate speaker · 1 min** shows two short passages to the on-camera person. Select a video source that shows the person while they read. Keep the speaker and camera position the same for later use. Calibration selects among existing decoder settings using the first passage, then accepts a change only if it improves the held-out passage; it does not update neural-network weights. A different speaker or source resets the profile. Keep the face large and front-facing, with the mouth unobstructed.

For screen capture on macOS, enable **Electron** in System Settings → Privacy & Security → Screen & System Audio Recording. If you launch with `npm start` from Terminal or iTerm2, enable that terminal app too, since macOS may attribute capture to the responsible launching app. The in-app settings link opens this pane; macOS requires you to add apps yourself. If Electron is not listed, click **+**, press **⌘⇧G**, and choose `node_modules/electron/dist/Electron.app` in this project. Quit and reopen Electron and the terminal after granting access. The Electron npm bundle is ad hoc signed; if capture remains denied, this development setup may need an Apple Development certificate and a properly signed app bundle. Opening a video file works without screen-capture permission. Keep the source video visible and playing. Protected playback may be unavailable to capture.

### Optional research sample

The UI's **Try a sample clip** button looks for `.cache/samples/speech.mp4`. A small LRS3 example published with the [SPECTRE research code](https://github.com/filby89/spectre/tree/master/samples/LRS3) can be downloaded for local testing:

```sh
mkdir -p .cache/samples
curl -fL https://raw.githubusercontent.com/filby89/spectre/master/samples/LRS3/QoT0S2vu9m4_00002.mp4 -o .cache/samples/speech-source.mp4
ffmpeg -i .cache/samples/speech-source.mp4 -an -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart .cache/samples/speech.mp4
```

This is an external research fixture, not our own training/evaluation corpus. The sample is excluded from Git and must not be treated as proof of general accuracy. Review source/dataset terms before redistribution.

The sample's original MPEG-4 Part 2 video codec is not supported by Electron's player, so the command converts it to H.264 and removes audio. For other unsupported files, use the same conversion command with your own input/output paths; do not overwrite your original video.

## Model assets and provenance

| Asset | Pinned value |
| --- | --- |
| Reference implementation | [mpc001/auto_avsr](https://github.com/mpc001/auto_avsr) |
| Git revision | `182b62837773ab01052d4ac21ef1d2203ea7d267` |
| Checkpoint | `vsr_trlrs2lrs3vox2avsp_base.pth`, English visual-only |
| Authors' download | [Google Drive model-zoo link](https://drive.google.com/file/d/1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_/view) |
| Local checkpoint | `.cache/models/autoavsr-en.pth` |
| SHA-256 | `fbf7cd70ff1c0e694b3030fb779dbb4570f04e4b841d62f9296c229e94878ddb` |
| MD5 | `49f770f2c0d8b8d769347ee47ed1648f` |

The optional VALLR visual-to-phoneme checkpoint is stored at `.cache/models/vallr.path`. `scripts/setup_vallr.py` pins the source revision and verifies SHA-256 `967667d61e705b0bc78d40a3ec80dd7ec449aa1d4d90a5e83c0d4289309ef374`. It is licensed CC BY-NC 4.0 and currently has no sentence-text decoder in the app.

`scripts/setup_model.py` verifies the full SHA-256 before installing/loading the downloaded checkpoint. The runtime uses `torch.load(..., weights_only=True)` and strict parameter matching. Face alignment follows the authors' MediaPipe preprocessing, followed by the published 88-pixel center crop, grayscale conversion, and normalization. The adapter imports the reference network directly and does not install its training framework.

Auto-AVSR code is Apache-2.0; its [license notes](https://github.com/mpc001/auto_avsr#license) say pretrained models may have separate terms derived from their training datasets. Keep the upstream license with its source. Deployment/redistribution rights for weights and datasets remain a release gate. This prototype does not establish commercial deployment rights.

The Auto-AVSR repository reports 20.3% WER on LRS3 for its visual model. VALLR reports 18.7% on LRS3 with its phoneme-to-sentence stage, which is not included in the downloaded visual checkpoint. Those benchmark numbers are not directly comparable on this app's streaming inputs. VALLR remains unavailable in the app until word decoding is integrated.

## Local recognition and current limits

- One session, one visible face, English, normal playback speed. Auto-AVSR produces sentence captions. VALLR assets are not exposed as a selectable model until word decoding is integrated.
- Native PyTorch MPS inference; no Docker or CUDA requirement. CPU is available for diagnostics, with no real-time guarantee.
- PyTorch 2.7.1 lacks MPS `max_pool3d`. The reference model's pool has temporal kernel/stride 1, so the MPS adapter replaces it with exactly equivalent framewise `max_pool2d`. A regression test compares the outputs exactly; no checkpoint parameters are changed.
- The desktop client crops the chosen video region, encodes frames at up to 960×720 as JPEG, and sends timestamped frames over authenticated loopback. **In this first version, face detection and mouth alignment run in the Python process on the same Mac**, rather than in a browser worker. Full-window frames never go to a remote service.
- Input is resampled to 25 fps using timestamps. The default accuracy profile completes approximately four-second intervals with an eight-path attention beam, CTC candidate rescoring, and a local GPT-2 language scorer. The existing CTC silence gate is preserved. Expensive provisional passes are disabled for this profile to keep inference caught up. Caption delay includes both collecting the interval and decoding it. The original four-path profile is available through `RML_DECODING_CONFIG=config/fast-decoder.json`.
- The optional one-minute speaker calibration compares attention beam, attention-greedy, and CTC-greedy decoding against two on-screen reading passages. The second passage is held out; the app only changes the in-memory decoder profile if it improves there. It never fine-tunes the model weights.
- Pending inference work is bounded. Overload emits an explicit skipped-interval message. Gaps, source changes, file seeks, and crop changes clear relevant context; old session/epoch results are ignored.
- The decoder has no calibrated confidence score. Face coverage gates missing/ambiguous faces but does not reliably detect nonspeech or all scene cuts. Even a clear face can produce incorrect captions.
- Interval boundaries can cut words. Reliable overlap reconciliation, speech/non-speech gating, automatic speaker tracking, and a sustained 30-minute evaluation remain follow-up work. No production accuracy or latency target is claimed yet.
- Captions currently appear in the app's preview and transcript, not in a separate always-on-top caption window.
- Screen capture, fullscreen, and hidden/minimized window behavior depend on macOS and the source application. Windows and Linux have not been validated.

## Verify and benchmark

```sh
npm test
npm run test:desktop
.venv/bin/python -m pip check
.venv/bin/python scripts/setup_model.py
.venv/bin/python scripts/benchmark.py .cache/samples/speech.mp4 --device mps --seconds 4
```

`npm test` runs TypeScript/Vite build checks and Python tests for timestamp resampling, bounded context, gaps, access control, epoch resets, short-clip flushing, and frame limits. Protocol tests use an explicitly labeled fake recognizer; the benchmark uses the real checkpoint and opens only the video stream.

`npm run test:desktop` launches the real Electron app, waits for the model, plays the installed sample, verifies completed recognition, mouth-input diagnostics, help, and the minimum window layout, and saves screenshots under `.cache/screenshots/`. It requires the optional sample and normal macOS GPU access. It reports screen-recording permission without changing it. `npm run test:desktop -- --capture` additionally verifies screen capture of a separate window containing only the public sample clip when permission is already granted.

The benchmark reports model-loading time, face coverage, preprocessing/decode time, device, and real-time factor (`processing seconds / video seconds`). A factor below one on one short clip is insufficient to prove sustained streaming or accuracy. Benchmark live capture, simultaneous playback, and longer sessions separately. MPS and MediaPipe graphics access can be unavailable inside a restricted command sandbox; run GPU/capture checks from a normal macOS terminal.

Use `--repeat 3` to distinguish initial GPU setup from warm inference. Optional `--decoder attention` and `--decoder ctc` compare greedy decoding variants; `--decoder hybrid` runs the slower reference attention+CTC beam as a diagnostic. The app defaults to the measured eight-path accuracy profile; `scripts/benchmark.py --decoder attention_beam` uses the loaded profile's beam configuration. Use `RML_DECODING_CONFIG=config/fast-decoder.json` to reproduce the original four-path behavior. On one 2.2-second MPS sample, attention beam took about 1.5 seconds and hybrid beam took about 14.4 seconds; this is a single-clip observation, not a sustained-streaming guarantee. See [docs/v1.md](docs/v1.md) for measured observations and remaining validation.

For the longer scheduling smoke test, create a repeated fixture and run:

```sh
ffmpeg -stream_loop -1 -i .cache/samples/speech.mp4 -t 20 -an -c copy .cache/samples/streaming.mp4
npm run test:desktop -- --dev --streaming
```

This is an integration stress fixture, not an accuracy dataset.

## TODO

Progress and measurements are in [docs/accuracy-report.md](docs/accuracy-report.md). The eight-path beam with CTC/local-LM rescoring is now the default after a development/evaluation comparison. Mouth-input diagnostics and evaluation/training tools are implemented. Overlap, alternative visual models, phoneme decoding, uncertainty thresholds, and a small adapter were tested; experimental changes that failed their checks remain outside the default path. Remaining research and real-capture validation are listed below.

- [x] **1. Establish an initial accuracy baseline (pilot complete; real-capture expansion pending).** Prepared 120 public English clips with dataset transcripts for the initial pilot, plus a separate 100-clip confirmation set. Human verification and representative recordings of the intended speaker, actual screen capture, varied head angles, and natural silent mouth/head motion remain follow-up work. Keep tuning and final evaluation recordings separate; hold out speakers when measuring generalization. Compare direct-file inference with the same clips played through screen capture to isolate transport/preprocessing damage. Report word error rate (WER), substitutions/deletions/insertions, false words per silent minute, caption coverage, p95 caption delay, and memory. This is a pilot dataset, not proof of broad accuracy.

- [x] **2. Audit and inspect the actual mouth input (reference audit and diagnostics complete).** Added an optional view of sampled actual grayscale mouth crops entering Auto-AVSR and audited the reference normalization/alignment. Landmark/tracking overlays and further tracking changes remain follow-up work. Compare crop geometry, RGB/grayscale handling, normalization, frame cadence, and interpolation against the checkpoint's reference pipeline. Track the chosen face across frames, smooth crop jitter without smoothing away lip movements, and detect occlusion or insufficient mouth resolution. Preserve the trained crop size and context. Existing alignment handles scale and in-plane rotation; severe yaw needs suitable training examples or an unreadable indication. Verify improvements with item 1 rather than assuming a tighter crop is better.

- [ ] **3. Stop cutting words at streaming boundaries (prototype tested; default remains off after boundary regressions).** Replace independent four-second intervals with overlapping context; initially compare 4–6-second windows with 0.5–1 second of overlap and endpoints near visual pauses. Merge overlapping hypotheses using timing/token alignment, commit only a stable prefix, and retain the uncertain suffix for revision. Measure missed and duplicated words, provisional revisions, and extra latency. Keep inference bounded and reject settings that cause a growing backlog on this Mac; overlap and more context cost compute and cannot guarantee zero delay.

- [x] **4. Tune decoding against visual evidence (initial sweep complete).** The original app used a four-path attention beam; the new default uses eight paths with measured CTC/LM rescoring weights. Sweep modest beam sizes (for example 4 and 8), attention/CTC score weights, length penalties, and end-of-sentence behavior on the tuning set. Retain multiple candidates and scores instead of immediately discarding all but the best path. Compare a practical hybrid decoder with the current attention beam; the existing hybrid implementation was much slower on our sample, so profile/batch its scoring before enabling it live. Preserve the working silence gate and include quiet speech when tuning it. A wider beam alone is not an accuracy guarantee.

- [ ] **5. Compare complete pretrained word-output models (AV-HuBERT and a combined decoder measured; MultiVSR downloads blocked).** Benchmark [MultiVSR](https://github.com/Sindhu-Hegde/multivsr), with its language fixed to English, and a visual-only fine-tuned [AV-HuBERT](https://github.com/facebookresearch/av_hubert) checkpoint against our Auto-AVSR baseline. Both projects publish checkpoints/inference paths, but download availability, preprocessing, older dependencies, MPS support, peak memory, and sustained streaming speed must be checked here. Use identical held-out clips and report accuracy and delay together; do not infer superiority from unrelated paper scores. Restore the model selector only for complete text-output pipelines that work locally.

- [x] **6. Add constrained language-model rescoring (implemented and evaluated).** Score the visual decoder's top 4–8 candidate sentences with a small local language model, combining language and visual scores with a weight tuned on held-out development data. Start with candidate reranking, optionally adding a modest user-supplied vocabulary bias for names or technical terms. Keep prior transcript context short and reset it on source/speaker changes. Measure whether the correct words are present among the candidates first: reranking cannot recover a missing candidate. Reject configurations that produce more plausible but unsupported words, increase silent insertions, or exceed the streaming budget.

- [ ] **7. Complete VALLR's phoneme-to-word stage (native-rate and lexicon prototypes failed the initial checks).** First audit its checkpoint-specific crop, pixel scaling, phoneme vocabulary, temporal sampling, and output sequence length; the current adapter samples only 16 frames per interval, and loading weights successfully does not establish useful recognition. Measure phoneme error rate on transcribed clips before adding a language model. Then compare a pronunciation-lexicon/CTC beam decoder with a compact learned phoneme-to-text decoder. Obtain compatible trained weights if available, or train the text stage on text-to-phoneme pairs using a pronunciation dictionary plus grapheme-to-phoneme conversion, augmented with realistic insertion/deletion/substitution errors measured from VALLR. Keep sentences/source documents separate across training and evaluation, and evaluate on real predicted phonemes as well as clean synthetic pairs. The upstream [LLM training script](https://github.com/MarshallT-99/VALLR/blob/main/Models/Llama.py) is a starting point, not a downloaded trained decoder. Only re-enable VALLR after measuring end-to-end word accuracy and latency. A fluent decoder cannot establish missing visual evidence.

- [ ] **8. Adapt pretrained weights to our actual video conditions (public GRID pilot and training pipeline complete; personal recordings needed).** Pilot a small, accurately transcribed collection of the intended speaker across separate sessions, lighting conditions, and moderate poses; start around 10–30 minutes and expand only if learning curves justify it. Mix it with broader speech examples to limit overfitting. Freeze most of the network and train a small adapter or limited decoder layers with a low learning rate, regularization, and early stopping. Use realistic compression, blur, crop jitter, and modest rotation augmentation. Accept changes only on new recordings and retain a general-speaker regression set. The existing one-minute calibration selects decoder settings; it is not weight training, and its passage must not become the accuracy test.

- [ ] **9. Calibrate uncertainty and allow abstention (margin audit did not meet accuracy/coverage targets; automatic threshold remains off).** Combine decoder score margins, CTC/attention disagreement, stability across overlapping windows, and crop quality to estimate when output is unreliable. Tune thresholds on development data and show an uncertain/unreadable interval instead of forcing a sentence. Keep silent, unreadable, and skipped intervals distinct. Report accuracy together with coverage and count abstentions as deletions in full-transcript WER, so hiding difficult speech cannot masquerade as improved recognition. Preserve silence clips as regression cases; the reported silence problem is already resolved.

- [ ] **10. Train a compact streaming model only after finding a better teacher (earlier compact GRID pilot regressed; broader training deferred).** Distill a validated word-output model into a smaller temporal encoder with CTC subword or phoneme outputs, using labeled clips and carefully filtered teacher predictions. Compare phoneme-plus-text decoding against direct subword decoding; lip positions alone do not uniquely identify sounds or words. Train on short chunks with bounded future context and realistic pauses. CTC blank is an alignment symbol, not a reliable semantic `[SILENCE]` token, so add supervised speech-activity/endpoint detection if needed. Pilot frozen-feature or adapter training on the Mac's MPS GPU and measure memory/time before scaling; do not assume the Apple Neural Engine is available through PyTorch for training. Training a full visual model from scratch is not the first local experiment.

For each experiment, record the checkpoint, preprocessing, decoder settings, dataset split, WER, silence errors, coverage, and latency. Promote a change only after a repeatable gain on untouched evaluation clips without an unacceptable streaming or silence regression. Keep all added Python packages in `requirements.txt` and document other tools and model/data setup here.

- [ ] Package and launch You're Not on Mute as its own signed macOS app without requiring a terminal. Screen Recording access should be granted to this app alone; requiring iTerm or Terminal access also permits screen capture for other processes launched from that terminal, which is a security risk.

## Project layout

```text
electron/               Desktop lifecycle, capture permissions, restricted preload
src/                    React workspace, playback, crop selection, frame transport
services/server.py      Local WebSocket API and bounded inference scheduling
services/inference/     Recognition, rescoring, alignment, evaluation, research adapters
config/                 Measured accuracy profile and original fast profile
public/                 Local brand icon
scripts/                Development launcher, verified model setup, real benchmark
tests/                  Temporal correctness and service protocol tests
.cache/                 Downloaded source/model/sample and local build caches (ignored)
.venv/                  Isolated Python environment (ignored)
```

When changing Python dependencies, update `requirements.txt` from the project environment after verification. Keep non-Python requirements and setup changes documented in this file.
