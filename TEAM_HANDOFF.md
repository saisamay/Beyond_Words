# Beyond Words | Engineering handoff
20 November 2026 · Implementation reviewed at [389e307](https://github.com/saisamay/Beyond_Words/commit/389e3079dfc7c7fe99b170a10383db0af776e935) on `feat/voice-model` · [Integration PR #1](https://github.com/saisamay/Beyond_Words/pull/1)

## 1. What a teammate is taking over

Beyond Words is a working local voice-companion prototype. A person records or uploads a short clip, gets a transcript, sees a tentative vocal cue, reads a Gemini reply, and can play that reply aloud. They can also compare how Gemini answers the same words with and without the cue. The product goal is a conversation that remembers what the person said and responds with appropriate wording and vocal delivery.

The verified milestone is **one live browser turn**. The current backend starts each turn with empty conversation history. Speech playback is available on demand. Persistent chat, progressive results, adaptive spoken delivery and a verified hosted service remain open. The app can call Google successfully while running on a laptop because the laptop hosts FastAPI and authenticates outbound Google API calls. A teammate visiting localhost connects to their own machine.

## 2. What runs where

| Component | Technology and code | Responsibility |
|---|---|---|
| Browser | HTML, CSS and JavaScript in [web/](web/) | Obtain consent, record up to 12 seconds, preview or download, upload and display results. Retain the current clip for retry. |
| HTTP and audio processing | Python 3.12, FastAPI, Uvicorn, ffmpeg in [app.py](app.py) | Enforce upload limits, normalize to mono 16 kHz PCM16 WAV, schedule transcription and cue inference concurrently, and assemble the response. |
| Vocal cue | FunASR, emotion2vec+ Base, NumPy and PyTorch in [predict_voice.py](predict_voice.py) | Extract a 768-dimensional embedding, apply the saved training mean and scale, classify with the 768 → 256 → 128 → 7 MLP, and decode with the checkpoint’s saved label map. |
| Transcription | Google Cloud Speech-to-Text in [companion.py](companion.py) | Transcribe the normalized audio. Short uploads use REST by default after repeated gRPC deadlines in local testing. |
| Reply | Vertex AI Gemini through google-genai 3.6.1 in [companion.py](companion.py) | Generate a short spoken-style reply using the transcript, tentative cue and system prompt. Record the resolved model and stage timings. |
| Speech playback | Google Cloud Text-to-Speech | Synthesize the selected reply to MP3 when the user requests playback. Language and voice are configurable. The playback request currently takes reply text only. |
| Packaging and checks | [Dockerfile](Dockerfile), [requirements.txt](requirements.txt), [tests/](tests/), [scripts/](scripts/) | Pin dependencies, package the CPU runtime, test endpoint contracts, and diagnose real provider failures separately. |

### What happens after Send

1. The browser posts the clip. The backend rejects empty, invalid, oversized or overlong uploads. Uploaded clips have an 8 MiB / 20-second limit.
2. After conversion, two workers start on the same WAV. One calls Google STT. The other uses the warmed local model. Startup loads the encoder and checkpoint once per healthy process. Inference access is serialized across concurrent callers.
3. Gemini starts when the transcript and available cue are ready. A cue failure still allows a transcript-only reply. An STT failure preserves the clip for retry or a clearly labeled manual transcript.
4. The browser waits for the complete turn response before displaying the transcript, cue and reply. This explains why a one-second inference can appear to take the entire turn. Playback adds a separate synthesis request.
5. The server removes temporary audio after the request. The clip remains in page memory until replaced or refreshed. The comparison token contains transcript data, expires after 15 minutes and becomes invalid on process restart. Keep recordings, tokens, diagnostic artifacts and credentials private.

## 3. How the implementation evolved

| Problem observed | Change and evidence |
|---|---|
| A Gemini candidate returned no speakable text | The original 512-token budget was exhausted, including 509 thinking tokens in one diagnostic. The backend uses LOW thinking and a configurable 2,048-token budget, extracts final text and rejects truncated output. |
| Provider returned 429 during a turn or comparison | The configured primary gets one SDK attempt with a default 12-second timeout. A structured capacity error can trigger one configured fallback attempt. A failed reply preserves transcript and cue. |
| Comparing replies from different models confused the experiment | The signed comparison captures the original settings and transcript. If cue-off needs a fallback, the backend regenerates cue-on with that same model and returns the matched pair. This controls model choice, although generation can still vary. |
| Repeated audio work made short clips feel slow | The backend warms the model, reuses a complete cached encoder, and overlaps transcription and inference. A compatible WAV can skip conversion. |
| The companion interpreted a sharp complaint as joking | Commit 389e307 changed the prompt to address the complaint directly, keep a steady tone and offer a useful next step. A cue can guide delivery without overriding the speaker’s words. Simulated checks confirmed prompt propagation through fallback and comparison. Listener acceptance of this revision is pending. |
| A locally running server produced a configuration error | The failing browser was on port 8080. The verified configured process was on 8090. A later restart failed because another process still held 8090. Use the explicit configuration below in the shell that starts the server. |

## 4. Run it on a teammate’s laptop

**First collect the private prerequisites.** Ask the model owner for the trusted classifier checkpoint and the complete encoder directory. Use an approved Google Cloud identity with project access, billing and the Speech-to-Text and Vertex AI APIs enabled. Enable Text-to-Speech for playback. Install Git, Python 3.12 with venv support, ffmpeg, libsndfile and the Google Cloud CLI. Chrome is useful for the browser smoke test.

### Get the working branch and dependencies

```bash
git clone --branch feat/voice-model https://github.com/saisamay/Beyond_Words.git
cd Beyond_Words
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip check
ffmpeg -version
```

For an existing checkout, run `git status --short --branch` first, switch to the feature branch when your worktree is ready, and use `git pull --ff-only origin feat/voice-model`. The most recent verified feature code includes commit 389e307.

### Supply the model assets and Google identity

Place the trusted classifier at `model_weights/emotion2vec_final_model.pth`, or export `CHECKPOINT_PATH` pointing to its actual location. The loader uses pickle-capable checkpoint loading. Obtain this file directly from the model owner. For a reproducible setup, point `EMOTION2VEC_MODEL_PATH` at the complete encoder snapshot. It needs model.pt, config.yaml, configuration.json and tokens.txt with their actual contents. The verified base revision was `b318240bfe67db81a8c572ecb37ce9c3759b81c9`.

```bash
# Replace this example path with the directory provided by the model owner.
export EMOTION2VEC_MODEL_PATH=/absolute/path/to/emotion2vec_plus_base
export GOOGLE_CLOUD_PROJECT=beyond-words-515310
export GOOGLE_CLOUD_LOCATION=global
gcloud auth application-default login
gcloud auth application-default set-quota-project "$GOOGLE_CLOUD_PROJECT"
```

Use credentials authorized for the project. Each teammate should authenticate individually. Keep weights, credentials, .env and clips outside version control. A plain Uvicorn launch reads exported environment variables. An ignored .env file alone does not configure the process.

### Start one server

```bash
# Flash-first profile for checking tone without waiting for a Neon capacity error.
export GEMINI_MODEL=gemini-3.5-flash
unset GEMINI_FALLBACK_MODEL
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8090
```

To reproduce the earlier two-model demo, set `GEMINI_MODEL=gemini-4-neon` and `GEMINI_FALLBACK_MODEL=gemini-3.5-flash` in that shell before launch. Model access depends on the project. Flash-first is a test profile. Its latency improvement has not been benchmarked on equivalent clips.

Open [the local app](http://localhost:8090/) once Uvicorn reports that it is listening. In a second terminal:

```bash
curl -fsS http://127.0.0.1:8090/health
curl -fsS http://127.0.0.1:8090/v1/status
```

Check that the model is loaded. Consent, record a short sentence, stop, then select **Send to Companion**. Verify the words, cue, reply and playback. Use **Compare without vocal cue** and check the paired models. Send a second clip and confirm the load count stays at one. To inspect the response and timings with an approved sample:

```bash
curl -sS -F audio=@local_audio/beyond-words-sample.webm \
  http://127.0.0.1:8090/v1/turn | python3 -m json.tool
```

The sample is private. Substitute a consented local recording if the sample is unavailable. Stop the server with Ctrl+C in its terminal. If 8090 is occupied, use `fuser -v 8090/tcp`, identify the old app process and terminate only that PID before restarting.

## 5. API contract for the frontend teammate

| Endpoint | Input | What the UI should do |
|---|---|---|
| GET /health and /v1/status | None | Use health for liveness. Use status for startup time and model load state. Verify cloud readiness with a real turn. |
| POST /v1/turn | Multipart audio, optional manual_transcript | Display transcript_source, transcript, vocal_cue and reply_text. Capture resolved_model and timings_ms for debugging. A 422 can invite manual retry. A 503 can still contain a usable transcript and cue. A 413 means upload too large. |
| POST /v1/compare | JSON comparison_token | Display cue_on_reply_text and cue_off_reply_text with their model IDs. Honor replace_original_reply when regeneration changes the baseline. |
| POST /v1/speak | JSON reply_text, up to 2,000 characters | Play returned MP3. Keep text visible and allow a playback retry if synthesis fails. |
| POST /v1/support-demo | Fictional support scenario provided by the server | Display the support reply and prepared handover brief. The demo does not create a ticket or contact a person. |

## 6. What has actually been verified

| Measurement or check | Recorded result | Interpretation |
|---|---|---|
| Live browser turn, private sample | Real Google transcript, trained neutral cue and successful Neon reply | Demonstrates one end-to-end turn with no manual transcript or provider doubles. |
| Cold startup / checkpoint load | 10.613 s / 2.510 s | Startup includes warmup. Two successive clips kept model load count at one. |
| Warm turn | 13.330 s Send-to-reply | Cue 1.022 s and STT 6.314 s overlapped. The parallel stage took 6.315 s, followed by 6.841 s for Gemini. One observation, not a service-level target. |
| Live comparison | Neon 429 after 7.052 s, Flash reply in 4.720 s, regenerated cue-on in 4.403 s | Both displayed comparison replies came from Flash. |
| Earlier STT failure | 25-second REST read timeout | Retrying the retained clip succeeded. Provider variability remains visible to the user. |
| Regression tests before prompt revision | 38 passed | Covered capacity failures, both-model failure, matched comparisons and parallel execution. Most cloud calls in regression tests use doubles. |
| Prompt revision and playback | Targeted simulated prompt/fallback checks passed. User demonstrated browser playback during local testing. | Re-run the full suite and listen to real post-revision replies before signing off on tone or voice quality. |

For regression work run `.venv/bin/python -m pytest -q`, then `.venv/bin/python -m tests.browser_smoke` with the required local sample and Chrome. Run `.venv/bin/python scripts/live_cloud_check.py` for independent service checks. A scripted Gemini probe can succeed while STT fails. For an end-to-end pass require the real transcript, checkpoint cue and generated reply from one request. Use [CHECKS.md](CHECKS.md) for dated evidence and [README.md](README.md) for diagnostics and configuration options. Earlier blocked probes describe earlier environments.

## 7. Where to look when something fails

| Symptom | First check |
|---|---|
| Transcript and cue appear, but Gemini asks for project or model | Verify the port in the address bar and the exported variables in the server shell. Restart that process. Flash fallback handles capacity errors only. |
| STT ReadTimeout | Retain the clip, retry, inspect transcription_ms and the sanitized service error. A typed retry should remain visibly labeled manual. |
| Neutral or surprising cue | Confirm model version and assets. Listen to the consented sample and compare the recognized words. A single predicted label cannot establish how the person feels. |
| Long pause before anything appears | Inspect prepare_ms, inference_ms, transcription_ms, audio_parallel_ms, gemini_ms and total_ms. Also check model attempts. The audio player’s duration measures playback length. |
| Works locally, only transcript works in a container | Check the mounted classifier, complete encoder files, read permissions and /v1/status. A successful health response alone is insufficient. |
| Comparison expires or playback fails | Obtain a fresh comparison token by sending a new turn. For playback, verify Text-to-Speech access and retry synthesis with the existing reply. |

## 8. Remaining work and suggested ownership

| Priority and owner | Work | Evidence for completion |
|---|---|---|
| 1 · Model + product | Evaluate actual tone and cue usefulness. Include frustration directed at the assistant, sadness, practical requests, neutral greetings, noisy clips and disagreements between wording and cue. Include relevant accents or mixed-language speech. | Reviewers listen to matched cue-on/off replies, flag mocking or patronizing wording, and record whether the response addressed the user’s actual concern. Track STT errors separately from cue and response errors. |
| 1 · Backend + frontend | Improve perceived and actual latency. Show transcription when available, stream reply text and investigate earlier playback. Measure before changing model order or transport. | Report median and p95 over repeated comparable clips, cold and warm separately. Track end-of-speech to visible transcript, first reply text and first audible reply. Count failures and fallback waits. |
| 1 · Backend + cloud | Deploy private staging using the repository’s Cloud Run plan. Stage trusted assets privately, mount read-only at /models, configure service identity, APIs, model IDs and region. | From an authenticated staging client, complete two real turns, inspect model version/load count, perform matched comparison and hear playback. Record deployed commit, resource use and timings. |
| 2 · Backend | Introduce session-scoped multi-turn context, then opt-in saved conversation history. Define ownership, retention, deletion and which turns are sent back to Gemini. | Three connected turns resolve a reference to the first turn. Two users’ histories stay isolated. Returning users can review and delete any saved history. |
| 2 · Voice + frontend | Evaluate pacing, pauses and voice choice on real replies. Add a deliberate mapping from conversation context to supported speech controls. Integrate recording, retry and playback into the teammate’s chat UI. | Listeners judge whether a reply sounds steady and natural. Casual questions get ordinary answers. Failed audio never hides the text reply. |
| 3 · Platform | Review concurrency, shared signing keys, provider quotas and resource limits before scaling beyond the demo. | Multi-instance comparisons survive routing changes. Load tests report queue times, memory, failures and recovery. |

The proposed staging configuration starts with one worker/instance, concurrency one, 2 CPUs and 4 GiB. These are starting settings for measurement. Comparison signing is process-local today. The README includes the draft deployment command and mount variables. The prior environment recorded a Docker daemon permission blocker. No hosted revision or production performance result was verified for this handoff.

**Suggested first team checkpoint:** Have a second teammate reproduce the local turn from a fresh checkout, capture timings, and independently rate a calm response. Use the same clip and acceptance checks in staging. Then build the three-turn memory test and progressive playback into the chat UI.
