"""Transparent scoring for dissonance, distress and next action."""
from fastapi.testclient import TestClient

import app
import companion
import insights


def test_masked_sarcasm_positive_words_negative_voice():
    cue = {"label": "disgusted", "scores": {"disgusted": 0.7, "angry": 0.2, "neutral": 0.1}}
    result = insights.build(cue, {"semantic_valence": 0.6, "distress": 10})
    assert result["vocal_valence"] == -0.65
    assert result["dissonance"] == "masked_sarcasm"
    assert result["discrepancy"] == 62  # |-0.65 - 0.6| / 2 * 100
    assert result["distress"] >= 50 and result["action"] == "de_escalate"


def test_congruent_calm_request_continues():
    cue = {"label": "neutral", "scores": {"neutral": 0.8, "happy": 0.2}}
    result = insights.build(cue, {"semantic_valence": 0.1, "distress": 5})
    assert result["dissonance"] == "congruent"
    assert result["action"] == "continue"


def test_high_distress_hands_over_to_a_human():
    cue = {"label": "angry", "scores": {"angry": 0.9, "sad": 0.1}}
    result = insights.build(cue, {"semantic_valence": -0.8, "distress": 90})
    assert result["dissonance"] == "congruent"
    assert result["distress"] >= 70 and result["action"] == "human_handover"


def test_without_a_cue_uses_words_only():
    result = insights.build(None, {"semantic_valence": -0.3, "distress": 45})
    assert result["vocal_valence"] is None and result["discrepancy"] is None
    assert result["dissonance"] == "words_only" and result["distress"] == 45


def test_analyze_text_endpoint_labels_scripted_voice(monkeypatch):
    monkeypatch.setattr(companion, "analyze_text", lambda t: {
        "semantic_valence": 0.7, "sentiment": "positive", "distress": 5, "intent": "praise"})
    monkeypatch.setattr(companion, "reply_result", lambda t, cue, **k: {
        "reply_text": "[TEST] I hear some frustration.", "resolved_model": "TEST", "attempts": [],
        "fallback_used": False, "gemini_ms": 1})
    response = TestClient(app.app).post("/v1/analyze-text", json={
        "transcript": "Great service, wonderful job.",
        "vocal_profile": {"label": "disgusted", "scores": {"disgusted": 0.8, "neutral": 0.2}}})
    data = response.json()
    assert response.status_code == 200
    assert data["vocal_source"] == "scripted"
    assert data["vocal_cue"]["model_version"] == "scripted-profile"
    assert data["insights"]["dissonance"] == "masked_sarcasm"
    assert data["reply_text"].startswith("[TEST]")
