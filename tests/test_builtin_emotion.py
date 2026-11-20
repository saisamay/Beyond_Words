"""The base-model fallback picks the strongest real emotion and skips 'other'/'<unk>'."""
import pytest

pytest.importorskip("funasr")
import predict_voice


def test_builtin_prediction_skips_non_emotions():
    labels = ["生气/angry", "开心/happy", "其他/other", "<unk>"]
    assert predict_voice._builtin_prediction(labels, [0.1, 0.2, 0.6, 0.1]) == {
        "label": "happy", "model_version": "emotion2vec_plus_base-builtin"}


def test_builtin_prediction_requires_a_real_emotion():
    with pytest.raises(ValueError):
        predict_voice._builtin_prediction(["其他/other", "<unk>"], [0.5, 0.5])
