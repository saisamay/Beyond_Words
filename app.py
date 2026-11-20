import base64
import hashlib
import hmac
import json
import logging
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, Response

import companion

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_BODY_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
MAX_AUDIO_SECONDS = 20
_prediction_lock = threading.Lock()
_token_key = secrets.token_bytes(32)
logger = logging.getLogger(__name__)


class UploadLimit:
    """Bound the entire multipart stream, including uploads without Content-Length."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        total = 0

        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > MAX_BODY_BYTES:
                raise HTTPException(413, "Upload too large. Use a clip under 8 MiB and 20 seconds.")
            return message

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    too_large = int(value) > MAX_BODY_BYTES
                except ValueError:
                    return await JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)(scope, receive, send)
                if too_large:
                    return await JSONResponse({"detail": "Upload too large. Maximum 8 MiB."}, status_code=413)(scope, receive, send)
        await self.app(scope, bounded_receive, send)

@asynccontextmanager
async def lifespan(application):
    started = time.perf_counter()
    try:
        from predict_voice import warm_model
        await run_in_threadpool(warm_model)
    except Exception as error:
        logger.warning("Startup vocal model unavailable: %s", type(error).__name__)
    application.state.startup_ms = round((time.perf_counter() - started) * 1000, 1)
    yield


app = FastAPI(title="Beyond Words", lifespan=lifespan)
app.add_middleware(UploadLimit)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def vocal_model_status() -> dict:
    module = sys.modules.get("predict_voice")
    return module.model_status() if module else {"loaded": False, "load_count": 0}


@app.get("/v1/status")
def status():
    return {"startup_ms": getattr(app.state, "startup_ms", None),
            "vocal_model": vocal_model_status()}


def predict(path: Path) -> dict:
    # A missing model dependency must not break health or transcript-only turns.
    from predict_voice import predict as model_predict

    with _prediction_lock:
        return model_predict(path)


def normalize(source: Path, target: Path) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
             "-t", str(MAX_AUDIO_SECONDS + 1), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(target)],
            check=True, capture_output=True, timeout=15,
        )
        with wave.open(str(target), "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
        if duration <= 0:
            raise ValueError("Empty audio")
        if duration > MAX_AUDIO_SECONDS:
            raise HTTPException(422, "Clip exceeds 20 seconds. Record a shorter clip.")
    except HTTPException:
        raise
    except (subprocess.SubprocessError, ValueError, wave.Error, EOFError):
        raise HTTPException(422, "Could not decode this clip. Record again or choose a valid audio file.")
    except FileNotFoundError:
        raise HTTPException(503, "Audio conversion is unavailable on the server (ffmpeg missing).")


# Speech-only cleanup: cut rumble/hiss, reduce steady background noise, and lift quiet
# speech to a consistent level. Used only for Speech-to-Text; the emotion model keeps the
# untouched clip so its training preprocessing is preserved.
SPEECH_FILTER = ("highpass=f=80,lowpass=f=7600,afftdn=nf=-25,"
                 "dynaudnorm=f=150:g=15:p=0.9:m=20,alimiter=limit=0.95")


def enhance_for_speech(wav: Path, target: Path) -> Path:
    """Return a speech-enhanced copy of ``wav``, or ``wav`` itself if enhancement fails."""
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(wav), "-af", SPEECH_FILTER,
             "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)],
            check=True, capture_output=True, timeout=15,
        )
        with wave.open(str(target), "rb") as enhanced:
            if enhanced.getnframes() > 0:
                return target
    except (subprocess.SubprocessError, OSError, wave.Error, EOFError):
        logger.warning("Speech enhancement unavailable; transcribing the original clip")
    return wav


def transcribe_with_retry(wav: Path) -> str:
    """Transcribe the enhanced clip; if nothing is recognized, retry on the original."""
    enhanced = enhance_for_speech(wav, wav.with_name("speech.wav"))
    try:
        return companion.transcribe(enhanced)
    except ValueError:
        if enhanced == wav:
            raise
        return companion.transcribe(wav)


def token_for(transcript: str, cue: dict | None, *, configuration: dict | None = None,
              reply_result: dict | None = None) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({
        "transcript": transcript, "vocal_cue": cue, "expires": time.time() + 900,
        "configuration": configuration if configuration is not None else companion.reply_configuration(),
        "original_reply": reply_result,
    }).encode()).decode()
    signature = hmac.new(_token_key, payload.encode(), hashlib.sha256).hexdigest()
    return payload + "." + signature


def read_token(token: str) -> dict:
    try:
        payload, signature = token.rsplit(".", 1)
        expected = hmac.new(_token_key, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError()
        data = json.loads(base64.urlsafe_b64decode(payload))
        if data["expires"] < time.time():
            raise ValueError()
        return data
    except (ValueError, KeyError, TypeError):
        raise HTTPException(422, "Comparison expired or invalid. Send the recording again.")


def timed_audio_call(function, path):
    started = time.perf_counter()
    try:
        return function(path), None, round((time.perf_counter() - started) * 1000, 1)
    except Exception as error:
        return None, error, round((time.perf_counter() - started) * 1000, 1)


def process_turn(source: Path, wav: Path, manual: str) -> dict:
    started = time.perf_counter()
    normalize(source, wav)
    timings = {"prepare_ms": round((time.perf_counter() - started) * 1000, 1)}
    fallbacks = []
    parallel_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="bw-audio") as pool:
        cue_future = pool.submit(timed_audio_call, predict, wav)
        text_future = pool.submit(timed_audio_call, transcribe_with_retry, wav) if not manual else None
        cue, cue_error, timings["inference_ms"] = cue_future.result()
        transcript, text_error, timings["transcription_ms"] = text_future.result() if text_future else (manual, None, 0)
    timings["audio_parallel_ms"] = round((time.perf_counter() - parallel_started) * 1000, 1)
    if cue_error:
        logger.warning("Vocal cue unavailable: %s", type(cue_error).__name__)
        fallbacks.append("vocal_cue_unavailable")
    transcript_source = "manual" if manual else "google_speech_to_text"
    if manual:
        fallbacks.append("manual_transcription")

    def finish(result):
        timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return {**result, "timings_ms": timings, "vocal_model": vocal_model_status(),
                "startup_ms": getattr(app.state, "startup_ms", None)}

    if text_error:
        logger.warning("Transcription unavailable: %s", type(text_error).__name__)
        return finish({"transcript": None, "transcript_source": None, "vocal_cue": cue,
                       "reply_text": None, "resolved_model": None,
                       "fallback_status": fallbacks + ["manual_transcript_required"],
                       "provider_errors": [{"service": "speech.googleapis.com", **companion.provider_error_details(text_error)}],
                       "error": "Transcription unavailable. Enter what you said and retry with this same clip."})
    configuration = companion.reply_configuration()
    gemini_started = time.perf_counter()
    try:
        result = companion.reply_result(transcript, cue, configuration=configuration)
    except Exception as error:
        logger.warning("Companion reply unavailable: %s", type(error).__name__)
        timings["gemini_ms"] = round((time.perf_counter() - gemini_started) * 1000, 1)
        errors = error.provider_errors if isinstance(error, companion.ReplyUnavailable) else [
            companion.provider_error_details(error, [transcript, companion.SYSTEM_PROMPT])]
        message = str(error) if isinstance(error, companion.ReplyUnavailable) else errors[0]["message"]
        return finish({"transcript": transcript, "transcript_source": transcript_source, "vocal_cue": cue,
                       "reply_text": None, "resolved_model": None,
                       "fallback_status": fallbacks + ["reply_unavailable"], "provider_errors": errors,
                       "error": "Companion unavailable. " + message})
    timings["gemini_ms"] = result["gemini_ms"]
    if result["fallback_used"]:
        fallbacks.append("gemini_fallback")
    token = token_for(transcript, cue, configuration=configuration, reply_result=result)
    return finish({"transcript": transcript, "transcript_source": transcript_source,
                   "vocal_cue": cue, "reply_text": result["reply_text"],
                   "resolved_model": result["resolved_model"], "gemini_attempts": result["attempts"],
                   "fallback_status": fallbacks, "comparison_token": token})


@app.post("/v1/turn")
async def turn(audio: UploadFile = File(...), manual_transcript: str = Form("", max_length=4000)):
    try:
        with tempfile.TemporaryDirectory(prefix="beyond-words-") as directory:
            source = Path(directory) / "recording"
            total = 0
            with source.open("wb") as destination:
                while chunk := await audio.read(64 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "Upload too large. Maximum 8 MiB.")
                    destination.write(chunk)
            if not total:
                raise HTTPException(422, "The clip is empty. Record some audio first.")
            result = await run_in_threadpool(process_turn, source, Path(directory) / "input.wav", manual_transcript.strip())
            status = 200 if result["reply_text"] else (503 if result["transcript"] else 422)
            return JSONResponse(result, status_code=status, headers={"Cache-Control": "no-store"})
    finally:
        await audio.close()


class Comparison(BaseModel):
    comparison_token: str = Field(max_length=16000)


@app.post("/v1/compare")
def compare(body: Comparison):
    data = read_token(body.comparison_token)
    if data["vocal_cue"] is None:
        raise HTTPException(422, "This turn already has no vocal cue. Send a clip with an available cue first.")
    original = data.get("original_reply")
    if not original:
        raise HTTPException(422, "Send a successful Companion turn before comparing.")
    started = time.perf_counter()
    attempts = []
    try:
        without = companion.reply_result(data["transcript"], None, configuration=data["configuration"],
                                          model=original["resolved_model"])
        attempts.extend({"arm": "cue_off", **item} for item in without["attempts"])
        regenerated = without["resolved_model"] != original["resolved_model"]
        if regenerated:
            with_cue = companion.reply_result(data["transcript"], data["vocal_cue"],
                configuration=data["configuration"], model=without["resolved_model"], allow_fallback=False)
            attempts.extend({"arm": "cue_on", **item} for item in with_cue["attempts"])
        else:
            with_cue = original
    except Exception as error:
        errors = error.provider_errors if isinstance(error, companion.ReplyUnavailable) else [
            companion.provider_error_details(error, [data["transcript"], companion.SYSTEM_PROMPT])]
        return JSONResponse({"detail": "Comparison unavailable. " + (
            str(error) if isinstance(error, companion.ReplyUnavailable) else errors[0]["message"]),
            "provider_errors": errors, "pair_ready": False}, status_code=503,
            headers={"Cache-Control": "no-store"})
    resolved = without["resolved_model"]
    display_text = without["reply_text"]
    if regenerated:
        # The current UI renders only reply_text. Include the regenerated baseline
        # here as well so it cannot silently compare the earlier Neon reply to Flash.
        # Dedicated fields let the frontend replace the original reply later.
        display_text += (f"\n\nMatched reply with vocal cue ({resolved}): {with_cue['reply_text']}"
                         f"\nBoth replies in this comparison use {resolved}. The earlier "
                         f"{original['resolved_model']} reply is not the comparison baseline.")
    return JSONResponse({"transcript": data["transcript"], "vocal_cue": None,
        "reply_text": display_text, "cue_off_reply_text": without["reply_text"],
        "cue_on_reply_text": with_cue["reply_text"], "resolved_model": resolved,
        "cue_on_model": with_cue["resolved_model"], "cue_off_model": resolved,
        "cue_on_regenerated": regenerated, "replace_original_reply": regenerated,
        "pair_ready": True, "conversation": data["configuration"]["conversation"],
        "generation_settings": data["configuration"]["settings"], "gemini_attempts": attempts,
        "cue_on_gemini_ms": with_cue["gemini_ms"], "cue_off_gemini_ms": without["gemini_ms"],
        "comparison_ms": round((time.perf_counter() - started) * 1000, 1),
        "fallback_status": ["gemini_fallback"] if regenerated or resolved == data["configuration"]["fallback_model"] else []},
        headers={"Cache-Control": "no-store"})


class SpeechRequest(BaseModel):
    reply_text: str = Field(min_length=1, max_length=2000)


@app.post("/v1/speak")
def speak(body: SpeechRequest):
    if not body.reply_text.strip():
        raise HTTPException(422, "Provide a reply to read aloud.")
    try:
        audio = companion.synthesize(body.reply_text)
    except Exception:
        raise HTTPException(503, "Google speech playback is unavailable. Your text reply is still available; retry playback later.")
    return Response(audio, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


@app.post("/v1/support-demo")
def support_demo():
    result = {**companion.SUPPORT_EXAMPLE, "demo": True, "vocal_cue": None,
              "transcript_source": "scripted_example"}
    try:
        result["reply_text"] = companion.support_reply()
    except Exception:
        return JSONResponse({**result, "reply_text": None,
                             "error": "Google reply unavailable. The fictional handover brief is still ready to review."},
                            status_code=503, headers={"Cache-Control": "no-store"})
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


app.mount(
    "/",
    StaticFiles(directory=Path(__file__).with_name("web"), html=True),
    name="web",
)
