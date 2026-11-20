"""Shared test setup: never call the real Gemini word analysis unless a test opts in."""
import pytest

import companion


@pytest.fixture(autouse=True)
def stub_word_analysis(monkeypatch, request):
    if "real_analysis" in request.keywords:
        return
    monkeypatch.setattr(companion, "analyze_text", lambda transcript: {
        "semantic_valence": 0.0, "sentiment": "neutral", "distress": 20, "intent": "test"})
