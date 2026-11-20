# Buildathon verification — 20 November 2026

## Backend handoff: bounded fallback and warm inference

Branch remains `feat/voice-model`; all existing work was preserved. No files in
`web/` changed during this backend task (content hashes checked). No new TTS work,
commit, push, or deployment was performed. The app is left running locally on
port 8090 for the user's manual check.

Implemented backend behavior:

- LOW thinking and the configurable 2048-token default are preserved.
- A structured Neon 429/RESOURCE_EXHAUSTED causes exactly one Flash attempt.
  Each model has one SDK attempt and a configurable 12000 ms SDK timeout.
- Successful results identify `resolved_model` and all sanitized attempts.
  Failure preserves transcript/cue and exact sanitized provider errors.
- Startup uses the existing complete encoder cache, loads the checkpoint once,
  and exposes the load count through `/v1/status`. STT and inference execute in
  parallel after audio preparation.
- Comparison snapshots transcript, system prompt, stateless context, settings,
  project/location and original resolved model. If cue-off falls back after an
  original Neon reply, cue-on is regenerated on Flash. No mismatched pair is
  returned. The existing UI receives both paired replies in its comparison
  text when regeneration is required; no UI source changes were needed.
- Repeated gRPC STT DeadlineExceeded errors were reproduced. A real REST STT
  request succeeded in 4.467 seconds. REST is now the default short-upload
  transport, with `SPEECH_TRANSPORT` preserving a configurable gRPC option.

Regression verification: **38 passed in 14.85s**. Simulated tests explicitly
cover Neon 429, one-shot fallback/retry limits, both-model failures, matching
comparison models, regeneration of Neon cue-on, no partial pair on failure,
frozen settings after environment changes, and concurrent audio stages.
These simulations are distinct from the real browser verification below.

Real browser verification used the supplied WebM, actual checkpoint, Google
STT, Neon/Flash, global, and ADC. No mocks or manual transcript were used.
An initial run completed two successive HTTP 200 Neon turns (18.405 s and
11.449 s), with load_count=1 for both. Its live comparison encountered a real
Neon 429, fell back to Flash, and regenerated cue-on on Flash successfully.

The final process and REST transport were then checked independently:

| Final verification | Evidence |
| --- | --- |
| Cold process startup | 10.6129 s, including actual checkpoint/encoder warmup |
| Checkpoint load | 2.5098 s; existing encoder cache used |
| Two successive clip inferences | Actual trained cue both times; load_count stayed at 1 |
| First REST browser request | STT ReadTimeout; 29.982 s to retryable error display |
| Warm REST browser request | HTTP 200; all three real outputs displayed; 13.330 s Send-to-reply |
| Warm inference / STT | 1.0223 s / 6.3142 s, concurrently; parallel stage 6.3149 s |
| Warm Neon reply | 6.8412 s; resolved model `gemini-4-neon` |
| Live cue-off comparison | Neon returned 429 after 7.0516 s; Flash replied in 4.7200 s |
| Regenerated cue-on | Flash replied in 4.4033 s; both pair model fields `gemini-3.5-flash` |
| Comparison display | Browser displayed the matched Flash pair and excluded the earlier Neon answer as its baseline |
| Browser JavaScript errors | None |

The final comparison preserved the same captured STT transcript, prompt,
conversation state and LOW/2048/temperature-0 settings. Private transcript and
reply content are intentionally omitted from this committed-document candidate;
the user-facing handoff reports the requested live output.

There is no blocker for the last successful live turn. A transient STT service
timeout remains possible; the exact failure observed on the preceding request
was:

```text
speech.googleapis.com — ReadTimeout
HTTPSConnectionPool(host='speech.googleapis.com', port=443): Read timed out. (read timeout=25.0)
```

The same retained clip succeeded on the next Send without manual transcription.
The live comparison also demonstrated the actual provider error handled by
fallback: `429 RESOURCE_EXHAUSTED: Resource exhausted. Please try again later.`

Files changed in this backend pass: `app.py`, `companion.py`, `predict_voice.py`,
`tests/test_fallback.py` (new), `tests/test_turn.py`,
`tests/test_google_adapters.py`, `tests/cloud_double_server.py`,
`scripts/compare_models.py`, `scripts/diagnose_gemini.py`, `README.md`, and
`CHECKS.md`. `.env`, credentials, recordings and model weights remain ignored;
no private assets are tracked. Compilation, dependency consistency, and diff
whitespace checks passed.

## Neon empty-reply diagnosis and correction

Installed SDK: `google-genai==3.6.1`. The original request used temperature 0,
512 output tokens, and provider-default thinking. Real sanitized reproduction:

```json
{
  "candidate_count": 1,
  "finish_reasons": ["MAX_TOKENS"],
  "prompt_block_reason": null,
  "part_types": [[{"types": ["text"], "thought": false}]],
  "token_usage": {
    "prompt_token_count": 205,
    "candidates_token_count": 3,
    "thoughts_token_count": 509,
    "total_token_count": 717
  }
}
```

No final text was present in either the SDK text accessor or candidate parts.
This was output exhaustion, not an extraction loss. The correction uses LOW
thinking and 2048 output tokens, requests a short spoken reply, extracts only
first-candidate final text, rejects truncated responses, and logs only
allowlisted metadata. Both settings remain configurable. No SDK upgrade needed.

Clean serial browser verification used the actual WebM, actual checkpoint,
Google STT, ADC, project `beyond-words-515310`, model `gemini-4-neon`, and location
`global`. No mocks or manual transcript were used. Earlier overlapping audit
samples were discarded; the table below contains only the fresh serial run.

| Measurement | Cold process/model | Warm model |
| --- | --- | --- |
| Actual checkpoint inference | 16.172 s | 0.931 s |
| Google STT | 5.237 s | 3.694 s |
| Gemini API time to provider error | 8.286 s | 9.210 s |
| Browser Send to error display | 30.248 s | 14.203 s |
| HTTP turn result | 503 | 503 |

The browser displayed the real transcript and trained cue in both cases. The
transcripts matched; private text is intentionally omitted from this document.
No Gemini reply was obtained. The exact provider error for both turns was:

```text
429 RESOURCE_EXHAUSTED
Resource exhausted. Please try again later. Please refer to https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429 for more details.
```

The independent cue-off request reused the captured warm STT transcript,
identical system prompt, empty conversation state, model/location, temperature,
LOW thinking, 2048-token budget, and timeout/retry settings. Only the cue was
removed. It returned the same provider 429 after 7.372 s. These are error timings,
not successful response latencies. Live verification of the correction and a
completed cue comparison remain blocked by Google capacity/quota.

Regression suite: **30 passed in 19.82s**. SDK unit tests use explicit doubles;
the browser checks above used real services. Tests cover truncated replies,
thought exclusion, candidate separation, blocked/empty responses, configurable
settings, metadata privacy, and redacted provider errors. Compilation,
`pip check`, and `git diff --check` passed. No commit, push or deployment occurred.

Fix-specific files ready for review: `companion.py`,
`scripts/diagnose_gemini.py`, `scripts/compare_models.py`,
`tests/test_google_adapters.py`, `README.md`, and `CHECKS.md`. The pre-existing
uncommitted MVP changes remain intact. Credentials, audio and model weights
remain ignored and are not part of the fix.

## Subsequent live Argon/Flash access check

ADC is now configured, and the user confirmed project `beyond-words-515310`.
The historical missing-ADC results below describe the initial MVP build.

Real live access requests used the same short scripted access prompt, existing
Companion system prompt and generation settings, ADC, the confirmed project,
and location `global`:

| Model | Reply | Time to API error | Result |
| --- | --- | --- | --- |
| `gemini-4-argon` | None | 2.911 s | 403 PERMISSION_DENIED / SERVICE_DISABLED |
| `gemini-3.8-flash` | None | 2.043 s | 403 PERMISSION_DENIED / SERVICE_DISABLED |

Google's exact message for both models was:

```text
Agent Platform API has not been used in project beyond-words-515310 before or it is disabled. Enable it by visiting https://console.developers.google.com/apis/api/aiplatform.googleapis.com/overview?project=beyond-words-515310 then retry. If you enabled this API recently, wait a few minutes for the action to propagate to our systems and retry.
```

The separate real Speech-to-Text request on the private sample returned:

```text
403 Cloud Speech-to-Text API has not been used in project beyond-words-515310 before or it is disabled. Enable it by visiting https://console.developers.google.com/apis/api/speech.googleapis.com/overview?project=beyond-words-515310 then retry. If you enabled this API recently, wait a few minutes for the action to propagate to our systems and retry.
```

These are error timings, not generation latency. No actual browser transcript
was produced, no same-turn model comparison completed, and no specificity or
cue-sensitivity improvement can be inferred. Argon was not promoted to the demo
default. Environment configuration remains intact and no APIs were enabled.
The new replay script `scripts/compare_models.py` captures one real turn and
holds its inputs fixed for both models once API access is available. Its
per-request model override does not mutate the server's configured model.
Validation after this change: `24 passed in 13.92s`; `git diff --check` passed.

## Initial MVP results

All changes remain uncommitted for review on the existing `feat/voice-model`
branch. Starting `git status --short` was empty; HEAD was `d24350f`, following
the starter commit. No branch switch, reset, commit, push, or deployment occurred.

## First-hour checks for the three roles

| Role | Check | Result | Evidence / limit |
| --- | --- | --- | --- |
| All three | Existing branch and work preserved | PASSED | `feat/voice-model`; existing commits retained |
| Model owner | Real inference on `local_audio/beyond-words-sample.webm` | PASSED | `neutral`, `emotion2vec_plus_base-mlp-v1`; initial CLI inference 3,289 ms |
| Model owner | Existing checkpoint/scaler/label map and base encoder | PASSED | Real local inference, real HTTP inference, and explicit local base-directory loading all returned `neutral` |
| Backend and cloud | `/health` on the running existing FastAPI app | PASSED | HTTP 200, `{"status":"ok"}` |
| Backend and cloud | Full local Companion turn with cloud test doubles | PASSED | Real browser clip, real decoder and trained inference; transcript and reply clearly labeled test doubles |
| Backend and cloud | Real Google Speech-to-Text | BLOCKED | ADC lookup raises `DefaultCredentialsError`; no real API response obtained |
| Backend and cloud | Real Vertex Gemini | BLOCKED | `GOOGLE_CLOUD_PROJECT` and `GEMINI_MODEL` unset; server configuration check raises `RuntimeError` before an API request |
| Backend and cloud | Docker build | BLOCKED | Docker client available; daemon connection at `/var/run/docker.sock` returns permission denied |
| Backend and cloud | Cloud Run deployment and hosted model inference | BLOCKED | `gcloud` absent, event project/credentials unconfigured, private checkpoint and base model not staged/mounted; deployment intentionally left for review |
| Frontend and product | Browser recording, stop and download | PASSED | Real Chrome MediaRecorder with fake microphone fed from the existing clip |
| Frontend and product | Browser playback | PASSED | Chrome playback currentTime advanced; human audibility and physical microphone remain unverified |
| Frontend and product | Physical microphone and audible output | BLOCKED | Headless automation has no physical user microphone/audibility verification |
| Frontend and product | Send, visible processing, retry, results | PASSED | Chrome exercised upload, real cue rendering, manual retry with unchanged selected clip, and text reply |
| Frontend and product | Compare without vocal cue | PASSED | Same transcript and Gemini prompt/settings; only cue becomes null; cloud output doubled |
| Frontend and product | Google TTS playback | BLOCKED | Missing ADC; SDK request and MP3 response contract tested with doubles, browser preserves text on actual credential failure |
| Frontend and product | Delayed-delivery support handover | PASSED | Fictional example and on-screen brief; support reply doubled; no ticket created/person contacted |

## Checks run

```text
.venv/bin/python -m pytest -q
23 passed in 11.99s

.venv/bin/python -m tests.browser_smoke
health, record/stop, playback, download: passed
real existing clip over HTTP: neutral
manual retry with same clip, comparison: passed
new recorded clip over HTTP: passed
TTS failure preserves reply, support handover: passed
JavaScript errors: []
Google transcript/reply/support outputs: explicitly labeled TEST DOUBLES

.venv/bin/python -m pip check
No broken requirements found.

node --check web/recorder.js
passed

git diff --check
passed
```

The endpoint suite checks absent/empty/invalid/oversized/overlong uploads,
requests without Content-Length, conversion format, manual transcription labels,
model failure with transcript-only reply, Gemini failure with preserved fields,
temporary file cleanup, comparison token validation/expiry, exact comparison
prompt inputs, TTS success/error contracts, and support fallback behavior.
Google boundary tests use actual SDK request/configuration types and fake API
clients. The private sample test uses real checkpoint inference.

An additional request to the production server, with no cloud doubles, returned:

```json
{
  "transcript": null,
  "transcript_source": null,
  "vocal_cue": {"label": "neutral", "model_version": "emotion2vec_plus_base-mlp-v1"},
  "reply_text": null,
  "fallback_status": ["manual_transcript_required"],
  "error": "Transcription unavailable. Enter what you said and retry with this same clip."
}
```

This was HTTP 422, with the server recording `DefaultCredentialsError` by class
name only. It is evidence for real inference and the fallback, not a completed
live Google turn. The independent `scripts/live_cloud_check.py` also reported
real cue `neutral`, STT blocked by missing ADC, and Gemini blocked by missing
configuration. No successful live Speech-to-Text, Gemini, or TTS call is claimed.

## Next live verification

Configure `GOOGLE_CLOUD_PROJECT`, `GEMINI_MODEL`, and `GOOGLE_CLOUD_LOCATION`
in the server shell, then establish event-authorized ADC on that same machine.
Restart the server and run:

```bash
.venv/bin/python scripts/live_cloud_check.py
curl -F audio=@local_audio/beyond-words-sample.webm http://localhost:8080/v1/turn
```

Report STT and Gemini separately and require a successful live HTTP response
containing all three real outputs. Test real TTS afterwards. The reviewed next
Cloud Run command, private asset paths, service identity, and container limits
are documented in [README.md](README.md). A container build alone is insufficient.

## Files changed

- `app.py`, `companion.py`, `predict_voice.py`: endpoints, Google adapters, existing inference integration.
- `web/index.html`, `web/recorder.js`, `web/recorder.css`: recorder/turn UI, retries, comparison, playback, support example.
- `requirements.txt`, `requirements-dev.txt`, `Dockerfile`: inference/cloud dependencies, test tools, CPU container runtime.
- `.gitignore`, `.gcloudignore`, `.dockerignore`: exclude private assets, credentials, recordings and artifacts.
- `tests/`: endpoint tests, real-model integration, Google SDK boundary tests and Chrome harness.
- `scripts/live_cloud_check.py`: separate real STT and Gemini verification after access is configured.
- `README.md`, `CHECKS.md`: local instructions, configuration names, private Cloud Run model plan and actual results.

Weights and recordings were verified ignored with `git check-ignore`.
The ignored browser screenshot is `artifacts/companion-browser.png`.
