"""Connect this entry point to the team's existing speech emotion model."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import TypedDict


class VocalPrediction(TypedDict):
    label: str
    model_version: str


def predict(audio_path: Path) -> VocalPrediction:
    """Apply training-time preprocessing and decode the result using its label map."""
    raise NotImplementedError("Connect predict() to your existing model and checkpoint.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path)
    audio_path = parser.parse_args().audio_path
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        parser.error("Provide a non-empty local audio file.")

    started = perf_counter()
    try:
        prediction = predict(audio_path)
    except NotImplementedError as error:
        parser.exit(1, f"{error}\n")
    elapsed_ms = round((perf_counter() - started) * 1000)
    print(json.dumps({**prediction, "elapsed_ms": elapsed_ms}, ensure_ascii=False))


if __name__ == "__main__":
    main()
