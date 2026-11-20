"""Connect this entry point to the team's existing speech emotion model."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from time import perf_counter
from typing import Optional, TypedDict

import numpy as np
import torch
import torch.nn as nn
from funasr import AutoModel

REPO_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = Path(os.environ.get("CHECKPOINT_PATH", REPO_ROOT / "model_weights" / "emotion2vec_final_model.pth"))
MODEL_VERSION = "emotion2vec_plus_base-mlp-v1"
# Used only when the team's trained checkpoint is absent (set EMOTION_BUILTIN_FALLBACK=0 to disable):
# emotion2vec+ base ships its own 9-class emotion head, so the cue still works without the private file.
BUILTIN_MODEL_VERSION = "emotion2vec_plus_base-builtin"
BUILTIN_LABELS = {"angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"}


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
_load_lock = threading.Lock()
_load_count = 0
_load_ms = 0.0
_encoder_source: Optional[str] = None
_builtin = False


def _cached_encoder_source() -> str:
    if configured := os.environ.get("EMOTION2VEC_MODEL_PATH"):
        return configured
    from huggingface_hub.constants import HF_HUB_CACHE
    root = Path(HF_HUB_CACHE) / "models--emotion2vec--emotion2vec_plus_base"
    revision = root / "refs" / "main"
    preferred = root / "snapshots" / revision.read_text().strip() if revision.is_file() else None
    snapshots = sorted((root / "snapshots").glob("*"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in ([preferred] if preferred else []) + snapshots:
        if all((path / name).is_file() for name in ("model.pt", "config.yaml", "configuration.json", "tokens.txt")):
            return str(path)
    return "iic/emotion2vec_plus_base"


def _loaded() -> bool:
    return _encoder is not None and (_classifier is not None or _builtin)


def model_status() -> dict:
    return {"loaded": _loaded(),
            "load_count": _load_count, "load_ms": _load_ms,
            "model_version": BUILTIN_MODEL_VERSION if _builtin else MODEL_VERSION,
            "trained_checkpoint": not _builtin and _classifier is not None,
            "encoder_cached": bool(_encoder_source and Path(_encoder_source).is_dir())}


def warm_model() -> dict:
    _ensure_loaded()
    return model_status()


def _ensure_loaded() -> None:
    if _loaded():
        return
    with _load_lock:
        _load_components()


def _builtin_fallback_enabled() -> bool:
    return os.environ.get("EMOTION_BUILTIN_FALLBACK", "1").strip().lower() not in ("0", "false", "no")


def _load_components() -> None:
    global _encoder, _classifier, _scaler_mean, _scaler_scale, _idx_to_emotion
    global _load_count, _load_ms, _encoder_source, _builtin

    if _loaded():
        return

    use_builtin = not CHECKPOINT_PATH.is_file()
    if use_builtin and not _builtin_fallback_enabled():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

    started = perf_counter()
    source = _cached_encoder_source()
    encoder = AutoModel(
        model=source,
        hub="hf",
        device="cpu",
        disable_update=True,
    )

    if use_builtin:
        _encoder = encoder
        _encoder_source = source
        _builtin = True
        _load_count += 1
        _load_ms = round((perf_counter() - started) * 1000, 1)
        return

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    model = EmotionClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    if mean.shape != (768,) or scale.shape != (768,):
        raise ValueError("Invalid saved scaler dimensions.")
    if not np.isfinite(scale).all() or np.any(scale == 0):
        raise ValueError("Invalid scaler values.")

    _scaler_mean = mean
    _scaler_scale = scale
    _idx_to_emotion = {int(k): v for k, v in checkpoint["idx_to_emotion"].items()}
    # Publish only after every component has loaded successfully; a retry is safe.
    _encoder = encoder
    _classifier = model
    _encoder_source = source
    _load_count += 1
    _load_ms = round((perf_counter() - started) * 1000, 1)


def _convert_to_wav(input_path: Path, output_path: Path) -> None:
    # The HTTP path already normalizes audio for STT. Preserve those exact samples.
    try:
        with wave.open(str(input_path), "rb") as wav:
            compatible = (wav.getnchannels(), wav.getframerate(), wav.getsampwidth(), wav.getcomptype()) == (1, 16000, 2, "NONE")
        if compatible:
            shutil.copyfile(input_path, output_path)
            return
    except (wave.Error, EOFError):
        pass
    command = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-v",
        "error",
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
        timeout=30,
    )


def predict(audio_path: Path) -> VocalPrediction:
    """Apply training-time preprocessing and decode the result using its label map."""
    _ensure_loaded()

    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "input.wav"
        _convert_to_wav(audio_path, wav_path)

        if _builtin:
            result = _encoder.generate(input=str(wav_path), granularity="utterance", extract_embedding=False)
            return _builtin_prediction(result[0]["labels"], result[0]["scores"])

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


def _builtin_prediction(labels, scores) -> VocalPrediction:
    """Pick the highest-scoring real emotion; labels look like '生气/angry'."""
    ranked = sorted(
        ((float(score), str(label).split("/")[-1].strip().lower()) for label, score in zip(labels, scores)),
        reverse=True,
    )
    for _, name in ranked:
        if name in BUILTIN_LABELS:
            return {"label": name, "model_version": BUILTIN_MODEL_VERSION}
    raise ValueError("No recognizable emotion in base model output.")


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
