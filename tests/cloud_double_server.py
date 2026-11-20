"""TEST ONLY: real inference, explicit cloud doubles; never use for the app run command."""
import companion
from app import app

_stt_calls = 0


def transcribe(path):
    global _stt_calls
    _stt_calls += 1
    if _stt_calls == 1:
        raise RuntimeError("TEST DOUBLE: exercise manual retry")
    return "[TEST DOUBLE transcript] I would like a little support today."


def reply(transcript, cue):
    return "[TEST DOUBLE reply] I'm here with you. What is one small thing that might help today?"


companion.transcribe = transcribe
companion.reply = reply
def reply_result(transcript, cue, **kwargs):
    model = kwargs.get("model") or "TEST-DOUBLE-MODEL"
    return {"reply_text": reply(transcript, cue), "resolved_model": model,
            "fallback_used": False, "gemini_ms": 1,
            "attempts": [{"model": model, "status": "passed", "elapsed_ms": 1}]}
companion.reply_result = reply_result
companion.support_reply = lambda: "[TEST DOUBLE support] A brief is ready for a person to review."
