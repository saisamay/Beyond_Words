"""Connect this entry point to the team's existing speech emotion model."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Optional, TypedDict

import numpy as np
import torch
import torch.nn as nn
from funasr import AutoModel

REPO_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = REPO_ROOT / "model_weights" / "emotion2vec_final_model.pth"
MODEL_VERSION = "emotion2vec_plus_base-mlp-v1"


class VocalPrediction(TypedDict):
    label: str
    model_version: str


class EmotionClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


_encoder: Optional[AutoModel] = None
_classifier: Optional[EmotionClassifier] = None
_scaler_mean: Optional[np.ndarray] = None
_scaler_scale: Optional[np.ndarray] = None
_idx_to_emotion: Optional[dict[int, str]] = None


def _ensure_loaded() -> None:
    global _encoder, _classifier, _scaler_mean, _scaler_scale, _idx_to_emotion

    if _encoder is not None and _classifier is not None:
        return

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

    _encoder = AutoModel(
        model="iic/emotion2vec_plus_base",
        hub="hf",
        device="cpu",
        disable_update=True,
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    model = EmotionClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _classifier = model

    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    if mean.shape != (768,) or scale.shape != (768,):
        raise ValueError("Invalid saved scaler dimensions.")
    if not np.isfinite(scale).all() or np.any(scale == 0):
        raise ValueError("Invalid scaler values.")

    _scaler_mean = mean
    _scaler_scale = scale
    _idx_to_emotion = {int(k): v for k, v in checkpoint["idx_to_emotion"].items()}


def _convert_to_wav(input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def predict(audio_path: Path) -> VocalPrediction:
    """Apply training-time preprocessing and decode the result using its label map."""
    _ensure_loaded()

    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "input.wav"
        _convert_to_wav(audio_path, wav_path)

        result = _encoder.generate(
            input=str(wav_path),
            granularity="utterance",
            extract_embedding=True,
        )

        embedding = np.asarray(result[0]["feats"], dtype=np.float32).reshape(1, -1)
        if embedding.shape != (1, 768):
            raise ValueError(f"Unexpected embedding shape: {embedding.shape}")
        if not np.isfinite(embedding).all():
            raise ValueError("Embedding contains NaN or Inf values.")

        embedding_scaled = (embedding - _scaler_mean) / _scaler_scale
        input_tensor = torch.from_numpy(embedding_scaled).float()

        with torch.no_grad():
            logits = _classifier(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        predicted_index = int(torch.argmax(probabilities).item())
        predicted_emotion = _idx_to_emotion[predicted_index].lower()

    return {
        "label": predicted_emotion,
        "model_version": MODEL_VERSION,
    }


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