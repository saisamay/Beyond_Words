# Beyond Words | Hour 1 starter

This starter serves a microphone recorder and a FastAPI health endpoint. The recorder keeps the clip in the browser and lets you download it for a local test. The model owner connects their existing checkpoint through `predict_voice.py`.

## Run locally

Prerequisites are Git, Python 3.10 or newer, and a browser with microphone access. Use a Python version compatible with your model in a separate environment if required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload --port 8080
```

Open [the recorder](http://localhost:8080/) and [the health endpoint](http://localhost:8080/health). The health response should be `{"status":"ok"}`. Record and download a five-second sample. On Windows PowerShell use `py -m venv .venv` and `.venv\Scripts\Activate.ps1`.

## Start with one shared Git repository

One teammate creates an empty private GitHub repository named `beyond-words` and grants the other two teammates access. Leave GitHub's README, license and gitignore initialization options unchecked when importing this starter.

From the extracted `beyond-words` directory, the repository owner runs these commands once. Replace `OWNER` with the GitHub username or organization.

```bash
git init -b main
git add .
git commit -m "Bootstrap Beyond Words"
git remote add origin https://github.com/OWNER/beyond-words.git
git push -u origin main
```

Each teammate clones the repository, creates their assigned branch, and pushes it after making a commit.

```bash
git clone https://github.com/OWNER/beyond-words.git
cd beyond-words
git switch -c feat/voice-model
# Work on your assigned files, then commit and push.
git add predict_voice.py
git commit -m "Connect existing vocal emotion inference"
git push -u origin HEAD
```

| Member | Branch | First responsibility |
| --- | --- | --- |
| Model owner | `feat/voice-model` | Run the existing model locally and connect the adapter |
| Backend and cloud | `feat/backend-cloud` | Verify health locally, then deploy the starter if the event project is ready |
| Frontend and product | `feat/voice-ui` | Capture a sample, confirm playback and download, and sketch Companion and Support |

Open a pull request into `main` once the local check passes. Share model weights through an approved private channel. Keep them in `model_weights/` and recordings in `local_audio/`, which are ignored by Git and the Cloud Run source upload. Keep credentials out of the repo.

## Connect your trained model

1. Reproduce one prediction using your original script or notebook, the original checkpoint, and an audio sample known to work.
2. Move the smallest working inference call into `predict(audio_path)` in `predict_voice.py`. Reuse the exact input format, sample rate, feature extraction, normalization, and label ordering from training.
3. Return `label` and `model_version`. Add per-class scores only if your inference code exposes them. Record the exact dependency versions the model requires.
4. Run the adapter on the known sample, then on a consented browser sample. Convert the browser audio only as needed to match the model. For example, use the following conversion only if the model expects mono 16 kHz PCM WAV.

```bash
ffmpeg -i local_audio/browser-clip.webm -ac 1 -ar 16000 -c:a pcm_s16le local_audio/model-input.wav
python predict_voice.py local_audio/model-input.wav
```

The command prints the actual adapter result and elapsed time. The starter deliberately exits with a clear error before you connect your model. The first timing may include model loading. For the eventual service, load the model once per process and measure warm inference separately.

## Verify Google Cloud access

Have one team member confirm the event's project and billing or credits. If permissions are ready, run a Cloud Run smoke deploy from the starter directory. Replace `YOUR_PROJECT_ID` with that project.

```bash
gcloud run deploy beyond-words --source . --project YOUR_PROJECT_ID --region asia-southeast1
```

Choose private access for the smoke deploy if prompted. Verify health through authenticated access or the Cloud Run console and share the URL or exact error with the team. Google AI calls are added after the existing model works locally.

## Product contract

One recording supplies two inputs. Your model reads the audio and provides a vocal cue. Google Speech-to-Text transcribes the same recording. The backend joins the cue and transcript for that turn and gives both to Gemini. Treat a model label as a tentative observation. An unavailable cue leaves the transcript usable.

Companion mode uses the cue to adapt response pacing, tone, and follow-up questions. Support mode adds business facts and explicit handover policy. Gemini drafts the reply in both modes. Google Text-to-Speech reads the approved response. GPT can optionally compare demo responses offline.
