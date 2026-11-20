"""Turn-level insights: vocal vs. semantic valence, dissonance, distress, and next action.

Pure functions only, so the scoring is transparent and unit-testable.

* Vocal valence (-1..1) comes from the emotion model's probabilities, weighted by a fixed
  valence per emotion (circumplex-style: happy is positive, angry/disgust/sad/fear negative).
* Semantic valence (-1..1) and word-level distress (0..100) come from Gemini reading only
  the literal words (see companion.analyze_text).
* Discrepancy (0..100) is how far apart the two valences are: |vocal - semantic| / 2 * 100.
* Distress (0..100) blends word distress (60%) with vocal negativity (40%).
"""

EMOTION_VALENCE = {
    "happy": 0.8,
    "surprised": 0.2,
    "neutral": 0.0,
    "sad": -0.6,
    "fearful": -0.6,
    "disgusted": -0.7,
    "angry": -0.8,
}

POSITIVE = 0.25  # valence beyond which words or voice count as clearly positive / negative


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def vocal_valence(cue: dict | None) -> float | None:
    """Probability-weighted valence of the voice, or None when no cue is available."""
    if not cue:
        return None
    scores = cue.get("scores") or {}
    known = {name: float(p) for name, p in scores.items() if name in EMOTION_VALENCE}
    if known and sum(known.values()) > 0:
        total = sum(known.values())
        return round(sum(EMOTION_VALENCE[name] * p for name, p in known.items()) / total, 3)
    label = str(cue.get("label", "")).lower()
    return EMOTION_VALENCE.get(label)


def classify(vocal: float | None, semantic: float) -> tuple[str, str]:
    """Return (kind, human explanation) for how the voice relates to the words."""
    if vocal is None:
        return "words_only", "No vocal cue — judged on the words alone."
    if semantic >= POSITIVE and vocal <= -POSITIVE:
        return "masked_sarcasm", "Positive words, negative voice — possible sarcasm or masked frustration."
    if semantic <= -POSITIVE and vocal >= POSITIVE:
        return "masked_distress", "Negative words, upbeat voice — the speaker may be downplaying a problem."
    if abs(vocal - semantic) >= 0.5:
        return "mismatch", "Voice and words point in different directions."
    return "congruent", "Voice and words agree."


def build(cue: dict | None, analysis: dict) -> dict:
    """Combine the vocal cue and Gemini's word analysis into one insight record."""
    semantic = round(clamp(float(analysis.get("semantic_valence", 0.0)), -1.0, 1.0), 3)
    word_distress = clamp(float(analysis.get("distress", 0.0)), 0.0, 100.0)
    vocal = vocal_valence(cue)
    kind, explanation = classify(vocal, semantic)
    discrepancy = None if vocal is None else round(abs(vocal - semantic) / 2 * 100)
    vocal_negativity = 0.0 if vocal is None else max(0.0, -vocal) * 100
    distress = round(word_distress if vocal is None else 0.6 * word_distress + 0.4 * vocal_negativity)
    if kind == "masked_sarcasm":
        distress = max(distress, 50)  # sarcasm about a service failure is never "all good"
    if distress >= 70:
        action, action_text = "human_handover", "Escalate to a human agent with a handover brief."
    elif distress >= 40 or kind in ("masked_sarcasm", "masked_distress"):
        action, action_text = "de_escalate", "Acknowledge the frustration, slow down, and offer a concrete next step."
    else:
        action, action_text = "continue", "Continue the conversation normally."
    return {
        "vocal_valence": vocal,
        "semantic_valence": semantic,
        "discrepancy": discrepancy,
        "dissonance": kind,
        "dissonance_text": explanation,
        "word_distress": round(word_distress),
        "distress": distress,
        "sentiment": analysis.get("sentiment"),
        "intent": analysis.get("intent"),
        "action": action,
        "action_text": action_text,
    }
