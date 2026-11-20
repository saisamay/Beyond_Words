"""SDK boundary tests use real Google request types, with fake API clients."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import companion


def response_with_text(text):
    from google.genai import types
    return types.GenerateContentResponse(candidates=[types.Candidate(
        finish_reason="STOP", content=types.Content(parts=[types.Part(text=text)]))])


def test_speech_request_uses_normalized_audio(tmp_path, monkeypatch):
    from google.cloud import speech

    path = tmp_path / "input.wav"
    path.write_bytes(b"TEST WAV")
    client = Mock()
    client.recognize.return_value = SimpleNamespace(results=[
        SimpleNamespace(alternatives=[SimpleNamespace(transcript="Some words.")])])
    factory = Mock(return_value=client)
    monkeypatch.setattr(speech, "SpeechClient", factory)
    assert companion.transcribe(path) == "Some words."
    assert factory.call_args.kwargs["transport"] == "rest"
    kwargs = client.recognize.call_args.kwargs
    assert kwargs["config"].sample_rate_hertz == 16000
    assert kwargs["config"].audio_channel_count == 1
    assert kwargs["config"].encoding == speech.RecognitionConfig.AudioEncoding.LINEAR16
    assert kwargs["audio"].content == b"TEST WAV"
    assert kwargs["retry"] is None
    assert kwargs["timeout"] == 25
    client.transport.close.assert_called_once()


def test_empty_speech_result_closes_client(tmp_path, monkeypatch):
    from google.cloud import speech

    path = tmp_path / "input.wav"
    path.write_bytes(b"TEST WAV")
    client = Mock()
    client.recognize.return_value = SimpleNamespace(results=[])
    monkeypatch.setattr(speech, "SpeechClient", lambda **kwargs: client)
    with pytest.raises(ValueError, match="No speech"):
        companion.transcribe(path)
    client.transport.close.assert_called_once()


def test_gemini_request_and_comparison_change_only_cue(monkeypatch):
    from google import genai

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "test-location")
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.models.generate_content.return_value = response_with_text("Test reply")
    factory = Mock(return_value=client)
    monkeypatch.setattr(genai, "Client", factory)
    transcript = "The exact same words."
    cue = {"label": "neutral", "model_version": "test"}
    assert companion.reply(transcript, cue) == "Test reply"
    assert companion.reply(transcript, None) == "Test reply"
    first, second = [call.kwargs for call in client.models.generate_content.call_args_list]
    assert first["model"] == second["model"] == "test-model"
    assert first["config"] == second["config"]
    assert first["config"].system_instruction == companion.SYSTEM_PROMPT
    assert first["config"].max_output_tokens == 2048
    assert first["config"].thinking_config.thinking_level.value == "LOW"
    assert first["contents"] == companion.prompt_data(transcript, cue)
    assert second["contents"] == companion.prompt_data(transcript, None)
    assert factory.call_args.kwargs["vertexai"] is True
    assert factory.call_args.kwargs["location"] == "test-location"
    assert factory.call_args.kwargs["http_options"].timeout == 12000


def test_tts_request_and_cleanup(monkeypatch):
    from google.cloud import texttospeech

    client = Mock()
    client.synthesize_speech.return_value = SimpleNamespace(audio_content=b"TEST MP3")
    monkeypatch.setattr(texttospeech, "TextToSpeechClient", lambda: client)
    assert companion.synthesize("A reply.") == b"TEST MP3"
    kwargs = client.synthesize_speech.call_args.kwargs
    assert kwargs["input"].text == "A reply."
    assert kwargs["audio_config"].audio_encoding == texttospeech.AudioEncoding.MP3
    assert kwargs["timeout"] == 25
    client.transport.close.assert_called_once()


def test_model_override_keeps_environment_and_prompt_fixed(monkeypatch):
    from google import genai

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_MODEL", "configured-demo-model")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.models.generate_content.return_value = response_with_text("Test reply")
    factory = Mock(return_value=client)
    monkeypatch.setattr(genai, "Client", factory)
    contents = companion.prompt_data("The same actual words.", {"label": "neutral", "model_version": "test"})
    for model in ("gemini-4-argon", "gemini-3.8-flash"):
        assert companion._generate(companion.SYSTEM_PROMPT, contents, model=model) == "Test reply"
    first, second = [call.kwargs for call in client.models.generate_content.call_args_list]
    assert first["model"] == "gemini-4-argon"
    assert second["model"] == "gemini-3.8-flash"
    assert first["contents"] == second["contents"] == contents
    assert first["config"] == second["config"]
    assert factory.call_args.kwargs["location"] == "global"
    import os
    assert os.environ["GEMINI_MODEL"] == "configured-demo-model"


def test_metadata_contains_structure_and_usage_without_private_content():
    from google.genai import types
    response = types.GenerateContentResponse(candidates=[types.Candidate(
        finish_reason="MAX_TOKENS", content=types.Content(parts=[
            types.Part(text="PRIVATE_THOUGHT", thought=True, thought_signature=b"PRIVATE_SIGNATURE"),
            types.Part(text="PRIVATE_REPLY"),
            types.Part(function_call=types.FunctionCall(name="PRIVATE_NAME", args={"private": "PRIVATE_ARGUMENT"})),
        ]))], usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=205, thoughts_token_count=509, candidates_token_count=3, total_token_count=717))
    metadata = companion.response_metadata(response)
    assert metadata["candidate_count"] == 1
    assert metadata["finish_reasons"] == ["MAX_TOKENS"]
    assert metadata["prompt_block_reason"] is None
    assert metadata["token_usage"]["thoughts_token_count"] == 509
    assert metadata["part_types"][0][0] == {"types": ["text"], "thought": True}
    import json
    assert "PRIVATE" not in json.dumps(metadata)
    assert companion.final_text(response) == "PRIVATE_REPLY"


def test_no_candidates_and_prompt_block_metadata():
    from google.genai import types
    response = types.GenerateContentResponse(prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY"))
    assert companion.final_text(response) == ""
    assert companion.response_metadata(response)["candidate_count"] == 0
    assert companion.response_metadata(response)["prompt_block_reason"] == "SAFETY"


def test_final_text_does_not_join_alternatives_or_thoughts():
    from google.genai import types
    response = response_with_text("First ")
    response.candidates[0].content.parts += [types.Part(text="PRIVATE_THOUGHT", thought=True), types.Part(text="reply.")]
    response.candidates += response_with_text("Another alternative.").candidates
    assert companion.final_text(response) == "First reply."


def test_generation_settings_are_configurable(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "medium")
    assert companion.generation_settings() == {"temperature": 0, "max_output_tokens": 4096, "thinking_level": "MEDIUM"}
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "MINIMAL")
    with pytest.raises(ValueError, match="GEMINI_THINKING_LEVEL"):
        companion.generation_settings()


def test_truncated_response_rejected_and_logs_are_private(monkeypatch, caplog):
    from google import genai
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-4-neon")
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    response = response_with_text("PRIVATE_INCOMPLETE_REPLY")
    response.candidates[0].finish_reason = "MAX_TOKENS"
    client.models.generate_content.return_value = response
    monkeypatch.setattr(genai, "Client", Mock(return_value=client))
    with pytest.raises(ValueError, match="MAX_TOKENS") as error:
        companion._generate("PRIVATE_PROMPT", "PRIVATE_TRANSCRIPT")
    assert "PRIVATE" not in str(error.value)
    assert "PRIVATE" not in caplog.text


def test_provider_error_redacts_private_inputs_and_credentials():
    class ProviderError(Exception):
        code = 429
        status = "RESOURCE_EXHAUSTED"
        message = "Capacity exhausted. PRIVATE_TRANSCRIPT Bearer secret_token ya29.test_access_token AIzaTestKey"
    details = companion.provider_error_details(ProviderError(), ["PRIVATE_TRANSCRIPT"])
    assert details["code"] == 429
    assert details["status"] == "RESOURCE_EXHAUSTED"
    assert "Capacity exhausted." in details["message"]
    assert "PRIVATE_TRANSCRIPT" not in details["message"]
    assert "secret_token" not in details["message"]
    assert "ya29." not in details["message"]
