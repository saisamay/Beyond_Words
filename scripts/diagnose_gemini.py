"""Reproduce Gemini responses with allowlisted diagnostics; never print audio or private text."""
import argparse
import json
import os
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import companion
from app import normalize, predict


def main():
    from google import genai
    from google.genai import types

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, default=Path("local_audio/beyond-words-sample.webm"))
    parser.add_argument("--legacy-settings", action="store_true", help="Reproduce the old 512-token request with default thinking")
    args = parser.parse_args()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    model = os.environ.get("GEMINI_MODEL")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
    if not project or not model:
        parser.error("Configure GOOGLE_CLOUD_PROJECT and GEMINI_MODEL")
    settings = companion.generation_settings()
    if args.legacy_settings:
        settings = {"temperature": 0, "max_output_tokens": 512}
    report = {"sdk_version": version("google-genai"), "model": model,
              "location": location, "settings": settings}
    private_values = [companion.SYSTEM_PROMPT]
    started = perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="bw-diagnose-") as directory:
            wav = Path(directory) / "clip.wav"
            normalize(args.clip, wav)
            transcript = companion.transcribe(wav)
            private_values.append(transcript)
            cue = predict(wav)
        config = types.GenerateContentConfig(
            system_instruction=companion.SYSTEM_PROMPT, temperature=settings["temperature"],
            max_output_tokens=settings["max_output_tokens"],
            thinking_config=types.ThinkingConfig(thinking_level=settings["thinking_level"])
            if "thinking_level" in settings else None,
        )
        with genai.Client(vertexai=True, project=project, location=location,
                          http_options=types.HttpOptions(timeout=companion.request_timeout_ms(), retry_options=types.HttpRetryOptions(attempts=1))) as client:
            result = client.models.generate_content(model=model,
                contents=companion.prompt_data(transcript, cue), config=config)
        report.update(metadata=companion.response_metadata(result),
                      final_text_present=bool(companion.final_text(result)))
    except Exception as error:
        report["provider_error"] = companion.provider_error_details(error, private_values)
    report["total_seconds"] = round(perf_counter() - started, 3)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("final_text_present") and "MAX_TOKENS" not in report["metadata"]["finish_reasons"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
