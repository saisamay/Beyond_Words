"""Replay identical Companion inputs against real Gemini models using ADC.

Private inputs, exact errors and results go to ignored artifacts/. No doubles.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import companion
from app import normalize, predict


def timed_request(model: str, contents: str) -> dict:
    started = perf_counter()
    try:
        reply = companion._generate(companion.SYSTEM_PROMPT, contents, model=model)
        return {"model": model, "status": "passed", "reply_text": reply,
                "elapsed_seconds": round(perf_counter() - started, 3)}
    except Exception as error:
        private_input = json.loads(contents)
        return {"model": model, "status": "failed", "error_type": type(error).__name__,
                "error": companion.provider_error_details(error, [private_input.get("transcript"), companion.SYSTEM_PROMPT]),
                "elapsed_seconds": round(perf_counter() - started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--models", nargs="+", default=["gemini-4-argon", "gemini-3.8-flash"])
    parser.add_argument("--clip", type=Path, default=Path("local_audio/beyond-words-sample.webm"))
    parser.add_argument("--turn-json", type=Path, help="Reuse an already captured real turn; no STT or inference repeat")
    parser.add_argument("--access-only", action="store_true", help="Scripted access check only; never a browser-turn comparison")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model-comparison.json"))
    args = parser.parse_args()
    if not args.project:
        parser.error("Supply --project or GOOGLE_CLOUD_PROJECT")
    os.environ["GOOGLE_CLOUD_PROJECT"] = args.project
    os.environ["GOOGLE_CLOUD_LOCATION"] = args.location
    report = {"project": args.project, "location": args.location,
              "calls": "real Google APIs using ADC; no test doubles",
              "timing": "single sequential wall-clock request per model, including client/auth/network; not a latency benchmark"}
    if args.access_only:
        turn = {"transcript": "This is a project access check. Please say hello in one brief sentence.",
                "vocal_cue": None, "transcript_source": "scripted_access_probe_not_browser_turn"}
    elif args.turn_json:
        turn = json.loads(args.turn_json.read_text())
        if not turn.get("transcript") or not isinstance(turn.get("vocal_cue"), dict):
            parser.error("The captured turn must contain a transcript and actual vocal_cue")
    else:
        with tempfile.TemporaryDirectory(prefix="bw-compare-") as directory:
            wav = Path(directory) / "clip.wav"
            normalize(args.clip, wav)
            try:
                transcript = companion.transcribe(wav)
                cue = predict(wav)
            except Exception as error:
                report.update(status="blocked", input_error_type=type(error).__name__, input_error=str(error))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2))
                print(json.dumps(report, indent=2), flush=True)
                return 1
            turn = {"transcript": transcript, "vocal_cue": cue,
                    "transcript_source": "google_speech_to_text", "clip": str(args.clip)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / "live-turn.json").write_text(json.dumps(turn, ensure_ascii=False, indent=2))
    contents = companion.prompt_data(turn["transcript"], turn["vocal_cue"])
    report.update(turn=turn, system_prompt=companion.SYSTEM_PROMPT, contents=contents,
                  fixed_input_sha256=hashlib.sha256((companion.SYSTEM_PROMPT + "\n" + contents).encode()).hexdigest(),
                  generation_settings={**companion.generation_settings(), "timeout_ms": companion.request_timeout_ms(), "attempts": 1},
                  results=[])
    for model in args.models:
        result = timed_request(model, contents)
        report["results"].append(result)
        summary = {key: value for key, value in result.items() if key != "reply_text"}
        summary["reply_present"] = bool(result.get("reply_text"))
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(r["status"] == "passed" for r in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
