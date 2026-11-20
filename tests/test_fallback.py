"""Explicit simulated provider failures. These are separate from live verification."""
import threading
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from google.genai import types

import app
import companion
from tests.test_turn import wav_bytes

NEON = "gemini-4-neon"
FLASH = "gemini-3.5-flash"
TEXT = "[SIMULATED STT] Can you help me take one small step?"
CUE = {"label": "neutral", "model_version": "SIMULATED-CUE"}


class CapacityError(Exception):
    code = 429
    status = "RESOURCE_EXHAUSTED"
    message = "Resource exhausted. Please try again later."


def response(text):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        finish_reason="STOP", content=types.Content(parts=[types.Part(text=text)]))])


@pytest.fixture(autouse=True)
def configuration(monkeypatch):
    for key, value in {"GOOGLE_CLOUD_PROJECT": "test-project", "GOOGLE_CLOUD_LOCATION": "global",
                       "GEMINI_MODEL": NEON, "GEMINI_FALLBACK_MODEL": FLASH,
                       "GEMINI_MAX_OUTPUT_TOKENS": "2048", "GEMINI_THINKING_LEVEL": "LOW",
                       "GEMINI_TIMEOUT_MS": "12000"}.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def sdk(monkeypatch):
    from google import genai
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    factory = Mock(return_value=client)
    monkeypatch.setattr(genai, "Client", factory)
    return client, factory


@pytest.fixture
def endpoint(monkeypatch):
    monkeypatch.setattr(app, "predict", lambda path: CUE)
    monkeypatch.setattr(companion, "transcribe", lambda path: TEXT)
    return TestClient(app.app)


def upload(client):
    return client.post("/v1/turn", files={"audio": ("simulated.wav", wav_bytes(), "audio/wav")})


def test_structured_429_falls_back_once_with_bounded_sdk_retries(sdk):
    client, factory = sdk
    client.models.generate_content.side_effect = [CapacityError(), response("[SIMULATED FLASH REPLY]")]
    result = companion.reply_result(TEXT, CUE)
    assert result["resolved_model"] == FLASH
    assert result["fallback_used"] is True
    assert [call.kwargs["model"] for call in client.models.generate_content.call_args_list] == [NEON, FLASH]
    for call in factory.call_args_list:
        assert call.kwargs["http_options"].timeout == 12000
        assert call.kwargs["http_options"].retry_options.attempts == 1
    configs = [call.kwargs["config"] for call in client.models.generate_content.call_args_list]
    assert configs[0] == configs[1]
    assert configs[0].thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert configs[0].max_output_tokens == 2048


def test_text_mentioning_429_does_not_trigger_fallback(sdk):
    client, _ = sdk
    client.models.generate_content.side_effect = ValueError("A message mentioning 429 RESOURCE_EXHAUSTED")
    with pytest.raises(companion.ReplyUnavailable):
        companion.reply_result(TEXT, CUE)
    assert client.models.generate_content.call_count == 1


def test_resource_exhausted_status_without_http_code_falls_back(sdk):
    client, _ = sdk
    class StatusError(Exception):
        status = "RESOURCE_EXHAUSTED"
    client.models.generate_content.side_effect = [StatusError(), response("[SIMULATED FLASH]")]
    assert companion.reply_result(TEXT, CUE)["resolved_model"] == FLASH


def test_both_models_fail_preserves_inputs_and_sanitized_errors(endpoint, sdk):
    client, _ = sdk
    client.models.generate_content.side_effect = [CapacityError(), CapacityError()]
    result = upload(endpoint)
    data = result.json()
    assert result.status_code == 503
    assert data["transcript"] == TEXT and data["vocal_cue"] == CUE
    assert data["reply_text"] is None and data["resolved_model"] is None
    assert [error["model"] for error in data["provider_errors"]] == [NEON, FLASH]
    assert all(error["code"] == 429 for error in data["provider_errors"])
    assert "Resource exhausted. Please try again later." in data["error"]
    assert TEXT not in data["error"]


def test_original_fallback_keeps_both_comparison_models_flash(endpoint, sdk):
    client, _ = sdk
    client.models.generate_content.side_effect = [CapacityError(), response("[SIMULATED FLASH ON]"),
                                                  response("[SIMULATED FLASH OFF]")]
    original = upload(endpoint).json()
    result = endpoint.post("/v1/compare", json={"comparison_token": original["comparison_token"]})
    data = result.json()
    assert result.status_code == 200
    assert original["resolved_model"] == data["cue_on_model"] == data["cue_off_model"] == FLASH
    assert data["cue_on_reply_text"] == original["reply_text"]
    assert not data["cue_on_regenerated"]
    assert [call.kwargs["model"] for call in client.models.generate_content.call_args_list] == [NEON, FLASH, FLASH]


def test_neon_original_is_regenerated_when_other_arm_needs_flash(endpoint, sdk, monkeypatch):
    client, _ = sdk
    client.models.generate_content.side_effect = [response("[SIMULATED NEON ON]"), CapacityError(),
        response("[SIMULATED FLASH OFF]"), response("[SIMULATED FLASH ON]")]
    original = upload(endpoint).json()
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "changed-after-original")
    data = endpoint.post("/v1/compare", json={"comparison_token": original["comparison_token"]}).json()
    assert original["resolved_model"] == NEON
    assert data["cue_on_model"] == data["cue_off_model"] == data["resolved_model"] == FLASH
    assert data["cue_on_reply_text"] == "[SIMULATED FLASH ON]"
    assert data["cue_off_reply_text"] == "[SIMULATED FLASH OFF]"
    assert data["cue_on_regenerated"] and data["replace_original_reply"]
    assert data["cue_on_reply_text"] in data["reply_text"]
    assert data["cue_off_reply_text"] in data["reply_text"]
    assert "not the comparison baseline" in data["reply_text"]
    assert data["conversation"] == []
    assert data["generation_settings"]["max_output_tokens"] == 2048
    calls = client.models.generate_content.call_args_list
    assert [call.kwargs["model"] for call in calls] == [NEON, NEON, FLASH, FLASH]
    assert all(call.kwargs["config"] == calls[0].kwargs["config"] for call in calls)
    import json
    bodies = [json.loads(call.kwargs["contents"]) for call in calls]
    assert all(body["transcript"] == TEXT for body in bodies)
    assert bodies[0]["tentative_vocal_cue"] == bodies[-1]["tentative_vocal_cue"] == CUE
    assert bodies[1]["tentative_vocal_cue"] is bodies[2]["tentative_vocal_cue"] is None


def test_no_partial_pair_when_flash_regeneration_fails(endpoint, sdk):
    client, _ = sdk
    client.models.generate_content.side_effect = [response("[SIMULATED NEON]"), CapacityError(),
        response("[SIMULATED FLASH OFF]"), CapacityError()]
    original = upload(endpoint).json()
    result = endpoint.post("/v1/compare", json={"comparison_token": original["comparison_token"]})
    assert result.status_code == 503
    assert result.json()["pair_ready"] is False
    assert "reply_text" not in result.json()


def test_stt_and_inference_are_concurrent(endpoint, sdk, monkeypatch):
    client, _ = sdk
    client.models.generate_content.return_value = response("[SIMULATED REPLY]")
    inference_started = threading.Event()
    stt_started = threading.Event()
    def predict(path):
        inference_started.set()
        assert stt_started.wait(timeout=2), "STT was not started in parallel"
        return CUE
    def transcribe(path):
        stt_started.set()
        assert inference_started.wait(timeout=2), "Inference was not started in parallel"
        return TEXT
    monkeypatch.setattr(app, "predict", predict)
    monkeypatch.setattr(companion, "transcribe", transcribe)
    result = upload(endpoint)
    assert result.status_code == 200
    assert result.json()["transcript"] == TEXT and result.json()["vocal_cue"] == CUE
