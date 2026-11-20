"""The base-model fallback picks the strongest real emotion and skips 'other'/'<unk>'."""
import pytest

pytest.importorskip("funasr")
import predict_voice


def test_builtin_prediction_skips_non_emotions():
    labels = ["生气/angry", "开心/happy", "其他/other", "<unk>"]
    result = predict_voice._builtin_prediction(labels, [0.1, 0.2, 0.6, 0.1])
    assert result["label"] == "happy"
    assert result["model_version"] == "emotion2vec_plus_base-builtin"
    # Renormalized over real emotions only.
    assert result["scores"] == {"happy": 0.6667, "angry": 0.3333}


def test_builtin_prediction_requires_a_real_emotion():
    with pytest.raises(ValueError):
        predict_voice._builtin_prediction(["其他/other", "<unk>"], [0.5, 0.5])
