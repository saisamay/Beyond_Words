"""Run with .venv/bin/python -m tests.browser_smoke. Google calls are test doubles.

Chrome records and plays a fake microphone fed from the real private browser clip.
The vocal model, HTTP uploads, ffmpeg, DOM, recording and playback are real.
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright, expect


def main():
    root = Path(__file__).resolve().parents[1]
    clip = root / "local_audio/beyond-words-sample.webm"
    with tempfile.TemporaryDirectory(prefix="bw-browser-") as directory:
        wav = Path(directory) / "microphone.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(clip),
                        "-ac", "1", "-ar", "16000", str(wav)], check=True)
        log = (Path(directory) / "server.log").open("w+")
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "tests.cloud_double_server:app",
                                   "--host", "127.0.0.1", "--port", "8081"], cwd=root, stdout=log, stderr=log)
        try:
            for _ in range(100):
                try:
                    if httpx.get("http://127.0.0.1:8081/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(.1)
            else:
                raise RuntimeError("Test server did not start")
            with sync_playwright() as p:
                browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                    args=["--no-sandbox", "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                          f"--use-file-for-fake-audio-capture={wav}", "--autoplay-policy=no-user-gesture-required"])
                context = browser.new_context(permissions=["microphone"])
                page = context.new_page()
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto("http://127.0.0.1:8081/")
                page.locator("#consent").check()
                page.locator("#clip").set_input_files(str(clip))
                page.locator("#send").click()
                expect(page.locator("#manual-panel")).to_be_visible(timeout=90000)
                expect(page.locator("#vocal-cue")).to_contain_text("neutral")
                page.locator("#manual").fill("[MANUAL TEST INPUT] I would like some support.")
                page.locator("#send").click()
                expect(page.locator("#reply")).to_contain_text("[TEST DOUBLE reply]", timeout=90000)
                expect(page.locator("#transcript-source")).to_contain_text("Manual transcription")
                page.locator("#compare").click()
                expect(page.locator("#comparison")).to_contain_text("Without vocal cue:", timeout=30000)
                expect(page.locator("#record")).to_be_enabled()
                page.locator("#record").click()
                expect(page.locator("#stop")).to_be_enabled()
                page.wait_for_timeout(2500)
                page.locator("#stop").click()
                expect(page.locator("#send")).to_be_enabled()
                page.locator("#playback").evaluate("audio => audio.play()")
                page.wait_for_timeout(600)
                assert page.locator("#playback").evaluate("audio => audio.currentTime > 0")
                with page.expect_download() as download:
                    page.locator("#download").click()
                assert download.value.suggested_filename.startswith("beyond-words-sample.")
                page.locator("#send").click()
                expect(page.locator("#reply")).to_contain_text("[TEST DOUBLE reply]", timeout=90000)
                expect(page.locator("#transcript")).to_contain_text("[TEST DOUBLE transcript]")
                expect(page.locator("#manual-panel")).to_be_hidden()
                page.locator("#speak").click()
                expect(page.locator("#status")).to_contain_text("speech playback is unavailable", timeout=30000)
                expect(page.locator("#reply")).to_contain_text("[TEST DOUBLE reply]")
                page.locator("summary").click()
                page.locator("#support-demo").click()
                expect(page.locator("#handover")).to_contain_text("No ticket created", timeout=30000)
                expect(page.locator("#support-reply")).to_contain_text("[TEST DOUBLE support]")
                assert not errors, errors
                artifacts = root / "artifacts"
                artifacts.mkdir(exist_ok=True)
                page.screenshot(path=str(artifacts / "companion-browser.png"), full_page=True)
                print(json.dumps({"health": "passed", "recording": "passed (fake microphone using real clip)",
                    "playback": "passed (headless time advances; human audibility unverified)",
                    "download": "passed", "real_clip_http_inference": "neutral",
                    "manual_retry_same_clip": "passed", "compare": "passed",
                    "recorded_clip_http_turn": "passed", "tts_failure_preserves_reply": "passed",
                    "support_handover": "passed", "cloud_calls": "TEST DOUBLES; real TTS blocked by missing ADC", "js_errors": errors}))
                browser.close()
        except Exception:
            log.flush()
            log.seek(0)
            print(log.read()[-4000:])
            raise
        finally:
            server.terminate()
            server.wait(timeout=15)
            log.close()


if __name__ == "__main__":
    main()
