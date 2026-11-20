"""Check the real Google services independently, without printing credentials.

Run .venv/bin/python scripts/live_cloud_check.py after server-side ADC setup.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import companion
from app import normalize, predict


def main():
    print("Configuration names:", json.dumps({name: bool(os.environ.get(name)) for name in (
        "GOOGLE_CLOUD_PROJECT", "GEMINI_MODEL", "GOOGLE_CLOUD_LOCATION")}))
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("local_audio/beyond-words-sample.webm")
    failures = []
    transcript = None
    cue = None
    with tempfile.TemporaryDirectory(prefix="bw-live-") as directory:
        wav = Path(directory) / "clip.wav"
        normalize(source, wav)
        try:
            cue = predict(wav)
            print("REAL VOCAL MODEL:", json.dumps(cue))
        except Exception as error:
            print("VOCAL MODEL FAILED:", type(error).__name__)
        try:
            transcript = companion.transcribe(wav)
            print("REAL SPEECH-TO-TEXT PASSED:", transcript)
        except Exception as error:
            failures.append("Speech-to-Text")
            print("REAL SPEECH-TO-TEXT BLOCKED/FAILED:", type(error).__name__)
        # A separate Gemini connectivity check still runs if STT fails. This is
        # explicitly scripted, never represented as a transcript from the clip.
        text = transcript or "[Scripted connectivity check, not a transcription] Please say hello briefly."
        try:
            reply = companion.reply(text, cue if transcript else None)
            print("REAL GEMINI PASSED:", reply)
            print("GEMINI INPUT SOURCE:", "real STT transcript + real cue" if transcript else "scripted connectivity check only")
        except Exception as error:
            failures.append("Gemini")
            print("REAL GEMINI BLOCKED/FAILED:", type(error).__name__)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
