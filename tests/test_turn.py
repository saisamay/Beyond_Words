import io
import json
import math
import struct
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app
import companion


@pytest.fixture
def client():
    return TestClient(app.app)


def wav_bytes(seconds=1):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"".join(struct.pack("<h", int(1000 * math.sin(i / 20))) for i in range(16000 * seconds)))
    return output.getvalue()


@pytest.fixture
def cloud_doubles(monkeypatch):
    paths = []
    calls = []

    def transcribe(path):
        paths.append(path)
        with wave.open(str(path)) as wav:
            assert wav.getframerate() == 16000
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
        return "[TEST DOUBLE] Today has been difficult."

    def reply(transcript, cue, **kwargs):
        calls.append((transcript, cue))
        model = kwargs.get("model") or "TEST-MODEL"
        return {"reply_text": "[TEST DOUBLE] That sounds difficult. Would taking one small step help?",
                "resolved_model": model, "fallback_used": False, "gemini_ms": 1,
                "attempts": [{"model": model, "status": "passed", "elapsed_ms": 1}]}

    monkeypatch.setattr(companion, "transcribe", transcribe)
    monkeypatch.setattr(companion, "reply_result", reply)
    return paths, calls


@pytest.fixture
def cue_double(monkeypatch):
    cue = {"label": "neutral", "model_version": "TEST-DOUBLE"}
    monkeypatch.setattr(app, "predict", lambda path: cue)
    return cue


def upload(client, content=None, **data):
    return client.post("/v1/turn", files={"audio": ("clip.wav", wav_bytes() if content is None else content, "audio/wav")}, data=data)


def test_health_and_static(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert "Send to Companion" in client.get("/").text
    assert client.get("/recorder.js").status_code == 200


@pytest.mark.parametrize("content", [b"", b"not audio"])
def test_invalid_audio(client, content):
    response = upload(client, content)
    assert response.status_code == 422
    assert "clip" in response.json()["detail"]


def test_missing_audio(client):
    assert client.post("/v1/turn").status_code == 422


def test_oversize(client):
    assert upload(client, b"x" * (app.MAX_UPLOAD_BYTES + 1)).status_code == 413
    assert client.post("/v1/turn", content=b"", headers={"content-length": str(app.MAX_BODY_BYTES + 1)}).status_code == 413


def test_chunked_oversize(client):
    def chunks():
        for _ in range(9):
            yield b"x" * (1024 * 1024)
    response = client.post("/v1/turn", content=chunks(), headers={"content-type": "multipart/form-data; boundary=test"})
    assert response.status_code in (400, 413)


def test_too_long(client):
    response = upload(client, wav_bytes(21))
    assert response.status_code == 422
    assert "20 seconds" in response.json()["detail"]


def test_turn_and_compare(client, cloud_doubles, cue_double):
    paths, calls = cloud_doubles
    response = upload(client)
    assert response.status_code == 200
    data = response.json()
    assert data["vocal_cue"] == cue_double
    assert data["fallback_status"] == []
    assert not paths[0].exists(), "Temporary recording must be removed"
    comparison = client.post("/v1/compare", json={"comparison_token": data["comparison_token"]})
    assert comparison.status_code == 200
    assert calls == [(data["transcript"], cue_double), (data["transcript"], None)]
    with_cue = json.loads(companion.prompt_data(*calls[0]))
    without = json.loads(companion.prompt_data(*calls[1]))
    assert with_cue.pop("tentative_vocal_cue") == cue_double
    assert without.pop("tentative_vocal_cue") is None
    assert with_cue == without


def test_unavailable_cue_still_replies(client, cloud_doubles, monkeypatch):
    def fail(path):
        raise FileNotFoundError("test missing model")
    monkeypatch.setattr(app, "predict", fail)
    response = upload(client)
    assert response.status_code == 200
    assert response.json()["vocal_cue"] is None
    assert response.json()["reply_text"]
    assert response.json()["fallback_status"] == ["vocal_cue_unavailable"]
    assert cloud_doubles[1][0][1] is None
    assert not cloud_doubles[0][0].exists()


def test_manual_retry_same_clip(client, cloud_doubles, cue_double, monkeypatch):
    paths = []
    def fail(path):
        paths.append(path)
        raise RuntimeError("test unavailable STT")
    monkeypatch.setattr(companion, "transcribe", fail)
    content = wav_bytes()
    response = upload(client, content)
    assert response.status_code == 422
    assert "manual_transcript_required" in response.json()["fallback_status"]
    assert not paths[0].exists()
    response = upload(client, content, manual_transcript="I typed this myself.")
    assert response.status_code == 200
    assert response.json()["transcript"] == "I typed this myself."
    assert response.json()["transcript_source"] == "manual"
    assert response.json()["fallback_status"] == ["manual_transcription"]
    assert len(paths) == 1


def test_gemini_failure_preserves_transcript(client, cloud_doubles, cue_double, monkeypatch):
    def fail(*args):
        raise RuntimeError("test unavailable Gemini")
    monkeypatch.setattr(companion, "reply_result", fail)
    response = upload(client)
    assert response.status_code == 503
    assert response.json()["transcript"]
    assert response.json()["reply_text"] is None
    assert response.json()["fallback_status"] == ["reply_unavailable"]
    assert not cloud_doubles[0][0].exists()


def test_manual_length_limit(client):
    assert upload(client, manual_transcript="x" * 4001).status_code == 422


def test_bad_comparison_token(client):
    assert client.post("/v1/compare", json={"comparison_token": "tampered"}).status_code == 422


def test_expired_token(client, monkeypatch):
    token = app.token_for("hello", {"label": "neutral"})
    monkeypatch.setattr(app.time, "time", lambda: 10**12)
    assert client.post("/v1/compare", json={"comparison_token": token}).status_code == 422


def test_real_browser_clip_inference_with_cloud_doubles(client, cloud_doubles):
    clip = Path("local_audio/beyond-words-sample.webm")
    if not clip.exists():
        pytest.skip("Private browser sample not available")
    response = upload(client, clip.read_bytes())
    assert response.status_code == 200
    data = response.json()
    assert data["vocal_cue"]["model_version"] == "emotion2vec_plus_base-mlp-v1"
    assert data["vocal_cue"]["label"] == "neutral"
    assert data["reply_text"].startswith("[TEST DOUBLE]")
    assert not cloud_doubles[0][0].exists()


def test_tts_double(client, monkeypatch):
    calls = []
    def synthesize(text):
        calls.append(text)
        return b"TEST DOUBLE MP3"
    monkeypatch.setattr(companion, "synthesize", synthesize)
    response = client.post("/v1/speak", json={"reply_text": "A supportive reply."})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == b"TEST DOUBLE MP3"
    assert calls == ["A supportive reply."]


def test_tts_failure_does_not_return_fake_speech(client, monkeypatch):
    def fail(text):
        raise RuntimeError("test unavailable TTS")
    monkeypatch.setattr(companion, "synthesize", fail)
    response = client.post("/v1/speak", json={"reply_text": "A supportive reply."})
    assert response.status_code == 503
    assert "text reply is still available" in response.json()["detail"]
    assert client.post("/v1/speak", json={"reply_text": " "}).status_code == 422
    assert client.post("/v1/speak", json={"reply_text": "x" * 2001}).status_code == 422


def test_support_example_and_handover(client, monkeypatch):
    monkeypatch.setattr(companion, "support_reply", lambda: "[TEST DOUBLE] A brief is ready for a person to review.")
    response = client.post("/v1/support-demo")
    data = response.json()
    assert response.status_code == 200
    assert data["demo"] is True
    assert data["vocal_cue"] is None
    assert data["transcript_source"] == "scripted_example"
    assert "No ticket created" in data["handover_brief"]["status"]


def test_support_cloud_failure_preserves_handover(client, monkeypatch):
    def fail():
        raise RuntimeError("test cloud unavailable")
    monkeypatch.setattr(companion, "support_reply", fail)
    response = client.post("/v1/support-demo")
    assert response.status_code == 503
    assert response.json()["handover_brief"]
    assert response.json()["reply_text"] is None
