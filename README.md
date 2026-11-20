# Beyond Words Buildathon MVP

The existing FastAPI app and browser recorder now support one Companion turn:
audio → trained emotion2vec + MLP cue and Google Speech-to-Text → Vertex Gemini
reply. The browser also offers a comparison without the cue, optional Google
Text-to-Speech, and a fictional delayed-delivery support example with a human
handover brief.

## Run in this workspace

Use Python 3.12, the existing `.venv`, and system `ffmpeg`:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8080
```

Open http://localhost:8080 and http://localhost:8080/health. Consent to sending
audio to the server and Google, 

, stop, listen, and select **Send to
Companion**. You can also choose the existing private sample. Missing or invalid
clips return useful errors. Clips are limited to 8 MiB and 20 seconds; the
recorder stops after 12 seconds.

```bash
.venv/bin/python predict_voice.py local_audio/beyond-words-sample.webm
curl -F audio=@local_audio/beyond-words-sample.webm http://localhost:8080/v1/turn
```

The real sample returned `neutral` using `emotion2vec_plus_base-mlp-v1`. See
[CHECKS.md](CHECKS.md) for the actual verification results and blockers.

## Server configuration

Environment variable names only; keep credentials and configuration values out
of Git and browser JavaScript:

| Variable | Purpose |
| --- | --- |
| `GOOGLE_CLOUD_PROJECT` | Event project ID; required for Vertex Gemini |
| `GEMINI_MODEL` | Event-approved Gemini model ID; required |
| `GEMINI_FALLBACK_MODEL` | Optional model tried once only after a structured 429 / RESOURCE_EXHAUSTED |
| `GEMINI_THINKING_LEVEL` | Optional thinking level; Companion defaults to LOW |
| `GEMINI_MAX_OUTPUT_TOKENS` | Optional combined output/thinking budget; defaults to 2048 |
| `GEMINI_TIMEOUT_MS` | Per-model SDK timeout; defaults to 12000 ms, one SDK attempt per model |
| `GOOGLE_CLOUD_LOCATION` | Vertex model location; configure for the approved model |
| `GOOGLE_APPLICATION_CREDENTIALS` | Optional ADC credential file path outside the repo; user ADC or Cloud Run identity normally avoids this |
| `GOOGLE_CLOUD_QUOTA_PROJECT` | Optional explicit quota project for Google client libraries |
| `SPEECH_LANGUAGE_CODE` | Optional Speech-to-Text language selection |
| `SPEECH_TRANSPORT` | Optional STT transport; short uploads use REST by default; gRPC remains selectable |
| `SPEECH_MODEL` | Optional STT recognition model; defaults to `latest_long` (more accurate on quiet/noisy speech), automatically retries with `default` if unsupported for the language |
| `SPEECH_PHRASES` | Optional comma-separated words/names to bias recognition (e.g. product names, order IDs) |
| `TTS_LANGUAGE_CODE` | Optional speech synthesis language |
| `TTS_VOICE_NAME` | Optional Google voice selection |
| `CHECKPOINT_PATH` | Optional classifier checkpoint override; default is the existing file in `model_weights/` |
| `EMOTION2VEC_MODEL_PATH` | Optional complete local base-model directory; required for private mounted model assets on Cloud Run |
| `HF_HOME` | Optional writable Hugging Face cache directory |
| `PORT` | Container listening port |

Export the confirmed project, Gemini model, and model location in the shell
that starts FastAPI. For local user ADC, install `gcloud`, then run:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project "$GOOGLE_CLOUD_PROJECT"
```

Restart FastAPI after setting environment variables. The event project needs
billing/credits and the Speech-to-Text and Vertex AI APIs enabled; optional
speech playback also needs Text-to-Speech. Use the event-approved identity and
permissions. No Gemini API key is required: Gemini uses Vertex AI and ADC.
Plain `uvicorn` does not automatically load `.env`; export the variables in the
server shell or supply them with `env` in the start command. Keep `.env` ignored.

After credentials are ready, run both real Google checks independently:

```bash
.venv/bin/python scripts/live_cloud_check.py
curl -F audio=@local_audio/beyond-words-sample.webm http://localhost:8080/v1/turn
```

The script reports Speech-to-Text and Gemini separately. If STT fails, its
separate Gemini connectivity probe uses clearly labeled scripted text; this is
not reported as an end-to-end turn. A complete live HTTP turn must return a real
transcript, the actual trained cue, and a real Gemini reply.

## Argon versus Flash replay

### Neon empty-reply correction

The installed `google-genai` SDK is 3.6.1. A real request with the original
512-token budget reproduced an empty reply with one candidate,
`finish_reason=MAX_TOKENS`, no prompt block, and 509 thinking tokens. Both the
SDK text accessor and candidate final text were empty; extraction did not lose
a hidden final reply. The Companion now requests LOW thinking with a 2048-token
budget and still asks for 1–3 short spoken sentences, at most 75 words.

The model and location remain environment-configurable. Truncated responses
are rejected rather than spoken as complete replies. Diagnostics allow only
candidate count, finish/block reasons, part type names, thought flags, and token
counters. They exclude prompts, replies, signatures, credentials and audio.

To reproduce with the real clip and print only sanitized diagnostics:

```bash
.venv/bin/python scripts/diagnose_gemini.py --legacy-settings
.venv/bin/python scripts/diagnose_gemini.py
```

The comparison script uses the same configurable settings in both arms and
prints only result summaries; private replies remain in its ignored result
artifact. A Google 429 is a provider capacity/quota error and is reported as
such; it does not establish whether the empty-reply correction succeeded live.

### Earlier model access check

`GEMINI_MODEL` and `GOOGLE_CLOUD_LOCATION` remain configurable. The requested
candidate is `gemini-4-argon` in `global`; the comparison model is
`gemini-3.8-flash`. No model is silently substituted, and no candidate is
promoted to a built-in demo default without successful access and comparison.

With the project environment configured, run:

```bash
.venv/bin/python scripts/compare_models.py --location global
```

This transcribes the existing clip once, runs the real trained cue once, then
replays identical transcript, cue, system prompt, temperature and output budget
through both real models. Only the model ID changes. The private captured turn,
exact error details, replies, input hash and request times go to ignored
`artifacts/`. To repeat without new transcription or inference:

```bash
.venv/bin/python scripts/compare_models.py --location global --turn-json artifacts/live-turn.json
```

Timing is a single sequential wall-clock measurement including client creation,
authentication and network time, not a latency benchmark. `--access-only` uses a
clearly labeled scripted access probe and does not claim to replay browser audio.

On 20 November, real ADC requests reached Google, but both model access probes
failed with `403 PERMISSION_DENIED`, reason `SERVICE_DISABLED`, for
`aiplatform.googleapis.com`. Speech-to-Text separately failed with the same
reason for `speech.googleapis.com`. No replies were obtained, so response quality
and generation latency cannot yet be compared. API enablement was not changed.
An authorized project administrator can resolve the prerequisite with:

```bash
gcloud services enable aiplatform.googleapis.com speech.googleapis.com --project "$GOOGLE_CLOUD_PROJECT"
```

After API enablement propagates, retry the real comparison before selecting the
demo model. Use `GEMINI_MODEL` to select the approved winner in the server shell.

## API and fallback behavior

- `GET /health`: liveness only; does not imply cloud or model readiness.
- `GET /v1/status`: process startup time and actual vocal-model load count,
  load time, model version, and whether the encoder came from the local cache.
- `POST /v1/turn`: multipart `audio`, optional `manual_transcript` (4,000 characters).
  Returns `transcript`, `transcript_source`, `vocal_cue`, `reply_text`, and
  `fallback_status`. Empty/invalid/missing clips return 422; oversized uploads
  return 413. Total request bytes are bounded even without Content-Length.
- A successful turn also returns `resolved_model`, sanitized `gemini_attempts`,
  `timings_ms`, `startup_ms`, and `vocal_model`. Neon and the configured fallback
  each get one SDK attempt. Only structured HTTP 429 or RESOURCE_EXHAUSTED causes
  fallback; text mentioning 429 does not. If both fail, HTTP 503 retains the
  transcript/cue and includes exact sanitized provider errors for each attempt.
- A cue failure still calls Gemini with the transcript alone. Transcription
  failure returns 422 with `manual_transcript_required`; the UI keeps the clip
  and lets the user type the words and retry the same recording. Manual input
  is always labeled `manual`. Gemini failure returns 503 with transcript/cue
  preserved, without inventing a reply.
- `POST /v1/compare`: JSON `comparison_token` from the successful turn. Reuses
  the exact transcript, system prompt, stateless conversation, project/location,
  and generation settings saved with the original turn. No new STT call. If the
  original used the fallback, cue-off uses that same model. If cue-off needs
  fallback after an original primary-model reply, cue-on is regenerated with
  the fallback too. A failed regeneration never returns an unmatched pair.
  `cue_on_model` and `cue_off_model` always match in a successful comparison.
  The response supplies `cue_on_reply_text`, `cue_off_reply_text`, and
  `replace_original_reply`. The current UI only renders `reply_text`, so when
  regeneration is needed that field includes both matched replies and clearly
  excludes the earlier primary-model answer from the comparison baseline.
  The frontend can later render the dedicated pair fields directly.
  Responses may still
  vary because model generation is not fully deterministic. The signed token
  expires after 15 minutes and includes the transcript; keep it private. This
  MVP uses one process/instance; restarts invalidate tokens.
- `POST /v1/speak`: JSON `reply_text` (up to 2,000 characters); returns Google
  MP3 audio. Failure preserves the text reply and is retryable.
- `POST /v1/support-demo`: fictional text input and order facts, no vocal cue,
  Gemini support reply, and an on-screen handover brief. Nothing is sent to a
  person or ticketing system. The brief remains usable when Gemini is unavailable.

The server warms the trusted checkpoint and encoder once during process startup,
preferring the complete local Hugging Face encoder cache before contacting the
model hub. `EMOTION2VEC_MODEL_PATH` still overrides that selection. Startup model
failure leaves transcript-only turns available. Subsequent recordings reuse the
same loaded components; `load_count` stays at one per healthy process.

The server decodes the clip to mono 16 kHz PCM16 WAV, then runs independent
STT and inference jobs concurrently. Already compatible WAV samples are copied without another
conversion. The original checkpoint architecture, scaler, embedding extraction,
and saved label map remain in use. Model initialization and inference are
serialized to avoid concurrent access. Temporary audio is removed after success
and failure; transcripts/audio are not logged by application code. Browser clips
remain only in page memory until refresh or replacement, unless downloaded.

## Local verification

```bash
.venv/bin/python -m pytest -q tests/test_turn.py
.venv/bin/python -m tests.browser_smoke
.venv/bin/python -m pip check
node --check web/recorder.js
git diff --check
```

Tests explicitly replace Google calls; the real sample test retains the real
checkpoint and encoder. The browser test launches installed Google Chrome and
a temporary test server, feeds the real sample to a fake microphone, and checks
record/stop/playback/download, HTTP inference, manual retry, comparison, TTS
failure, and support handover. It does not verify a physical microphone or human
audibility. Its screenshot is ignored at `artifacts/companion-browser.png`.
`tests.cloud_double_server` is test-only; never use it as the normal app command.

## Cloud Run review plan — do not deploy yet

The Dockerfile includes ffmpeg, libsndfile, the CPU inference dependencies, and
all app modules. It runs as a non-root user. Checkpoints, recordings, credentials,
archives, and local caches are excluded from Git, Docker, and source uploads.
A container build or `/health` response alone does not verify hosted inference.

Privately stage these two assets in an event-controlled Cloud Storage bucket:

1. `emotion2vec_final_model.pth`, the trusted trained MLP checkpoint with scaler
   and label map. It uses pickle loading, so only the team's trusted file is valid.
2. A fully materialized copy of the existing cached emotion2vec base snapshot,
   under `emotion2vec_plus_base/`, including `model.pt`, `config.yaml`,
   `configuration.json`, and `tokens.txt`. Dereference cache symlinks when copying;
   uploading symlinks without their target blobs will break inference. The
   locally verified base snapshot revision is
   `b318240bfe67db81a8c572ecb37ce9c3759b81c9`.

Do not publish weights or depend on the developer's home cache in the container.
Mount the private bucket read-only at `/models`. Set `CHECKPOINT_PATH` to the
mounted checkpoint and `EMOTION2VEC_MODEL_PATH` to the mounted base directory.
The path override lets FunASR load the complete local base instead of contacting
a public model hub. Keep its cache writable in `/tmp`.

Use a dedicated Cloud Run service identity with read access only to the model
bucket plus the event-approved Speech-to-Text and Vertex permissions. Cloud Run
obtains ADC from this identity; no credential JSON belongs in the image. See
Google's [Cloud Storage volume configuration](https://docs.cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts)
and [service identity guide](https://docs.cloud.google.com/run/docs/configuring/services/service-identity).

Once project access, the private bucket assets, service identity, and image are
reviewed, the next deployment command is below. Shell variables are deliberately
unfilled. This has **not** been executed:

```bash
gcloud run deploy beyond-words \
  --source . \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --region "$CLOUD_RUN_REGION" \
  --service-account "$CLOUD_RUN_SERVICE_ACCOUNT" \
  --no-allow-unauthenticated \
  --execution-environment gen2 \
  --memory 4Gi --cpu 2 --concurrency 1 --max-instances 1 --timeout 180 \
  --add-volume "mount-path=/models,type=cloud-storage,bucket=$MODEL_BUCKET,readonly=true" \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT,GEMINI_MODEL=$GEMINI_MODEL,GOOGLE_CLOUD_LOCATION=$GOOGLE_CLOUD_LOCATION,CHECKPOINT_PATH=/models/emotion2vec_final_model.pth,EMOTION2VEC_MODEL_PATH=/models/emotion2vec_plus_base"
```

Use one instance and one worker for this demo because comparison signatures are
process-local. After deployment, verify privately through `gcloud run services
proxy` and upload the consented sample through the proxy. Require the expected
`model_version` in the returned cue, not merely a transcript-only fallback.
Confirm cold-start memory and inference latency using the mounted assets.
Container build verification is currently blocked by Docker daemon permission;
hosted inference remains unverified until the private assets are mounted.

## Team first-hour responsibilities

| Role | Existing branch assignment | Verification |
| --- | --- | --- |
| Model owner | `feat/voice-model` | Real sample inference and preserved training preprocessing/labels |
| Backend and cloud | `feat/backend-cloud` | Local health and turn/fallback checks; event cloud access pending |
| Frontend and product | `feat/voice-ui` | Recorder, playback, send/retry/results, comparison and support example |

Work was performed on the existing `feat/voice-model` branch, preserving its
commits. No branch switch, commit, push, deployment, or external message was made.
