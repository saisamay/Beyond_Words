/**
 * Emotion Companion — drop-in page for the BeyondWords sign-language site.
 *
 * Talks to the Beyond Words voice backend (FastAPI): speech-to-text, vocal emotion cue,
 * and a Gemini companion reply. Requires: react, lucide-react, Tailwind CSS.
 * Configure the backend URL with the Vite env var VITE_COMPANION_API_URL.
 */
import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle, Bot, GitCompare, Loader2, Mic, MicOff, RotateCcw, Send, Square, Upload, User, Volume2,
} from "lucide-react";

const API = (import.meta.env.VITE_COMPANION_API_URL || "").replace(/\/$/, "");
const MAX_SECONDS = 12;

const EMOTIONS = {
  neutral: { emoji: "😐", label: "Neutral", cls: "text-slate-300 bg-slate-400/10 border-slate-400/30", dot: "bg-slate-400" },
  happy: { emoji: "😊", label: "Happy", cls: "text-emerald-300 bg-emerald-400/10 border-emerald-400/30", dot: "bg-emerald-400" },
  sad: { emoji: "😢", label: "Sad", cls: "text-sky-300 bg-sky-400/10 border-sky-400/30", dot: "bg-sky-400" },
  angry: { emoji: "😠", label: "Angry", cls: "text-red-300 bg-red-400/10 border-red-400/30", dot: "bg-red-400" },
  fearful: { emoji: "😨", label: "Fearful", cls: "text-purple-300 bg-purple-400/10 border-purple-400/30", dot: "bg-purple-400" },
  surprised: { emoji: "😮", label: "Surprised", cls: "text-amber-300 bg-amber-400/10 border-amber-400/30", dot: "bg-amber-400" },
  disgusted: { emoji: "🤢", label: "Disgusted", cls: "text-teal-300 bg-teal-400/10 border-teal-400/30", dot: "bg-teal-400" },
};

const NOTICES = {
  vocal_cue_unavailable: "Vocal emotion unavailable — reply based on your words alone.",
  gemini_fallback: "Primary model busy — answered by the fallback model.",
  manual_transcription: "Using your typed transcription.",
  manual_transcript_required: "We couldn't make out the words. Type what you said and send again.",
  reply_unavailable: "Companion reply unavailable right now. Your clip is kept — try again.",
};

async function postJson(path, body) {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Request failed.");
  return data;
}

function EmotionChip({ cue }) {
  if (!cue) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-2.5 py-0.5 text-xs font-medium text-slate-400">
        <span aria-hidden="true" className="text-base leading-none">🔇</span> Unavailable
      </span>
    );
  }
  const style = EMOTIONS[cue.label] || EMOTIONS.neutral;
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border py-0.5 pl-1.5 pr-3 text-sm font-semibold ${style.cls}`}>
      <span aria-hidden="true" className="text-lg leading-none">{style.emoji}</span> {style.label}
    </span>
  );
}

export default function EmotionCompanion() {
  const [consent, setConsent] = useState(false);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [clip, setClip] = useState(null); // { blob, url, name }
  const [busy, setBusy] = useState("");
  const [status, setStatus] = useState("Turn on consent to begin.");
  const [result, setResult] = useState(null);
  const [manual, setManual] = useState("");
  const [needsManual, setNeedsManual] = useState(false);
  const [comparison, setComparison] = useState(null);
  const [replyAudio, setReplyAudio] = useState(null);
  const [level, setLevel] = useState(Array(28).fill(0));

  const recorderRef = useRef(null);
  const timerRef = useRef(null);
  const meterRef = useRef(null);
  const audioCtxRef = useRef(null);

  useEffect(() => () => {
    clearInterval(timerRef.current);
    cancelAnimationFrame(meterRef.current);
    audioCtxRef.current?.close().catch(() => {});
  }, []);

  function loadClip(blob, name) {
    if (clip?.url) URL.revokeObjectURL(clip.url);
    setClip({ blob, url: URL.createObjectURL(blob), name });
    setResult(null);
    setComparison(null);
    setNeedsManual(false);
    setManual("");
    setReplyAudio(null);
  }

  function stopMeter() {
    cancelAnimationFrame(meterRef.current);
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
    setLevel(Array(28).fill(0));
  }

  async function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setStatus("Recording needs a modern browser on HTTPS.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      const chunks = [];
      const recorder = new MediaRecorder(stream, { audioBitsPerSecond: 128000 });
      recorder.ondataavailable = event => event.data.size && chunks.push(event.data);
      recorder.onstop = () => {
        stream.getTracks().forEach(track => track.stop());
        clearInterval(timerRef.current);
        stopMeter();
        setRecording(false);
        if (!chunks.length) return setStatus("No audio captured. Try again.");
        const type = chunks[0].type || recorder.mimeType || "audio/webm";
        loadClip(new Blob(chunks, { type }), `companion.${type.includes("ogg") ? "ogg" : type.includes("mp4") ? "m4a" : "webm"}`);
        setStatus("Recording ready — play it back or send it to the Companion.");
      };
      recorderRef.current = recorder;
      recorder.start();
      setRecording(true);
      setElapsed(0);
      setStatus(`Listening… speak naturally (up to ${MAX_SECONDS} s).`);
      const started = Date.now();
      timerRef.current = setInterval(() => {
        const seconds = Math.floor((Date.now() - started) / 1000);
        setElapsed(seconds);
        if (seconds >= MAX_SECONDS) recorder.state === "recording" && recorder.stop();
      }, 200);
      try {
        const context = new (window.AudioContext || window.webkitAudioContext)();
        const analyser = context.createAnalyser();
        analyser.fftSize = 64;
        context.createMediaStreamSource(stream).connect(analyser);
        audioCtxRef.current = context;
        const data = new Uint8Array(analyser.frequencyBinCount);
        const draw = () => {
          analyser.getByteFrequencyData(data);
          setLevel(Array.from({ length: 28 }, (_, i) => data[i % data.length] / 255));
          meterRef.current = requestAnimationFrame(draw);
        };
        draw();
      } catch { /* meter is decorative */ }
    } catch {
      setStatus("Microphone blocked. Allow microphone access and try again.");
    }
  }

  function stopRecording() {
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
  }

  async function send() {
    if (!clip || busy) return;
    setBusy("send");
    setStatus("Transcribing your words and reading your tone…");
    setComparison(null);
    setReplyAudio(null);
    try {
      const form = new FormData();
      form.append("audio", clip.blob, clip.name);
      if (needsManual && manual.trim()) form.append("manual_transcript", manual.trim());
      const response = await fetch(`${API}/v1/turn`, { method: "POST", body: form });
      const data = await response.json().catch(() => ({}));
      if (!response.ok && !("transcript" in data)) throw new Error(data.detail || "Could not process this clip.");
      setResult(data);
      const manualNeeded = (data.fallback_status || []).includes("manual_transcript_required") || data.transcript_source === "manual";
      setNeedsManual(manualNeeded);
      setStatus(data.error || "Companion reply ready.");
    } catch (error) {
      setStatus(`${error.message} Check that the Companion server is running.`);
    } finally {
      setBusy("");
    }
  }

  async function speak() {
    if (!result?.reply_text || busy) return;
    setBusy("speak");
    try {
      const response = await fetch(`${API}/v1/speak`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reply_text: result.reply_text }),
      });
      if (!response.ok) throw new Error("Speech playback unavailable.");
      const url = URL.createObjectURL(await response.blob());
      setReplyAudio(url);
      new Audio(url).play().catch(() => {});
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy("");
    }
  }

  async function compare() {
    if (!result?.comparison_token || busy) return;
    setBusy("compare");
    try {
      const data = await postJson("/v1/compare", { comparison_token: result.comparison_token });
      setComparison({ with: data.cue_on_reply_text, without: data.cue_off_reply_text ?? data.reply_text, model: data.resolved_model });
      setStatus("Comparison ready.");
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy("");
    }
  }

  const mmss = `00:${String(elapsed).padStart(2, "0")}`;
  const notices = (result?.fallback_status || []).map(key => NOTICES[key]).filter(Boolean);
  const total = result?.timings_ms?.total_ms;

  return (
    <div className="mx-auto w-full max-w-5xl space-y-6 px-4 py-6 text-slate-100">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-white/10 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Emotion Companion</h1>
          <p className="mt-1 text-sm text-slate-400">
            Speak naturally — we transcribe your words, read the emotion in your voice, and reply in a way that fits both.
          </p>
        </div>
        <span className="rounded-full border border-violet-500/30 bg-violet-500/10 px-3 py-1 font-mono text-[11px] uppercase tracking-widest text-violet-300">
          AI · Voice · Emotion
        </span>
      </header>

      <div className="grid gap-6 lg:grid-cols-5">
        {/* Capture */}
        <section className="space-y-5 rounded-2xl border border-white/10 bg-white/[0.03] p-6 lg:col-span-2">
          <label className="flex cursor-pointer items-start gap-3 text-sm text-slate-400">
            <input
              type="checkbox"
              className="mt-0.5 h-4 w-4 accent-violet-500"
              checked={consent}
              disabled={recording || !!busy}
              onChange={event => {
                setConsent(event.target.checked);
                setStatus(event.target.checked ? "Ready — tap the microphone." : "Turn on consent to begin.");
              }}
            />
            I agree to record and send this clip to the server and Google for this demo.
          </label>

          <div className="flex flex-col items-center gap-5 rounded-xl border border-white/10 bg-black/30 px-4 py-8">
            <div className="flex h-12 items-center gap-1" aria-hidden="true">
              {level.map((value, index) => (
                <span
                  key={index}
                  className={`w-1 rounded-full transition-all duration-75 ${recording ? "bg-violet-400" : "bg-white/15"}`}
                  style={{ height: `${Math.max(4, value * 48)}px` }}
                />
              ))}
            </div>
            <div className={`font-mono text-3xl font-semibold ${recording ? "text-red-400" : "text-slate-100"}`}>
              {mmss}
            </div>
            <div className="flex items-center gap-4">
              <button
                type="button"
                onClick={recording ? stopRecording : startRecording}
                disabled={!consent || !!busy}
                aria-label={recording ? "Stop recording" : "Start recording"}
                className={`grid h-20 w-20 place-items-center rounded-full text-white shadow-lg transition disabled:cursor-not-allowed disabled:opacity-40 ${
                  recording
                    ? "animate-pulse bg-red-500 shadow-red-500/30"
                    : "bg-gradient-to-br from-indigo-500 to-violet-600 shadow-violet-600/30 hover:scale-105"
                }`}
              >
                {recording ? <Square className="h-7 w-7" /> : consent ? <Mic className="h-8 w-8" /> : <MicOff className="h-8 w-8" />}
              </button>
            </div>
            <p className="text-center text-sm text-slate-400">
              {recording ? "Tap to stop" : consent ? "Tap to start recording" : "Turn on consent to record"}
            </p>
          </div>

          {clip && <audio controls src={clip.url} className="w-full [color-scheme:dark]" />}

          <label className={`flex cursor-pointer items-center gap-3 rounded-xl border border-dashed border-white/15 px-4 py-3 text-sm text-slate-400 transition hover:border-violet-400/60 hover:bg-violet-500/5 ${recording || busy ? "pointer-events-none opacity-50" : ""}`}>
            <Upload className="h-4 w-4 shrink-0" />
            <span><span className="font-semibold text-slate-200">Upload a clip</span> — up to 8 MB / 20 s</span>
            <input
              type="file"
              accept="audio/*,video/webm,video/mp4"
              className="hidden"
              onChange={event => {
                const file = event.target.files?.[0];
                if (!file) return;
                if (file.size > 8 * 1024 * 1024) return setStatus("Choose a clip smaller than 8 MB.");
                loadClip(file, file.name);
                setStatus(`“${file.name}” ready.`);
              }}
            />
          </label>

          {needsManual && (
            <div className="space-y-2 rounded-xl border border-amber-400/30 bg-amber-400/5 p-4">
              <p className="text-sm font-semibold text-amber-200">What did you say?</p>
              <textarea
                rows={3}
                maxLength={4000}
                value={manual}
                onChange={event => setManual(event.target.value)}
                placeholder="Type your words, then send again with the same clip."
                className="w-full rounded-lg border border-white/10 bg-black/30 p-3 text-sm text-slate-100 outline-none focus:border-violet-400"
              />
            </div>
          )}

          <button
            type="button"
            onClick={send}
            disabled={!clip || !consent || !!busy || recording}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-600 py-3 font-semibold text-white shadow-lg shadow-violet-600/20 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {busy === "send" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            {busy === "send" ? "Processing…" : "Send to Companion"}
          </button>
          <p className="text-center text-xs text-slate-500" role="status">{status}</p>
        </section>

        {/* Conversation */}
        <section className="flex min-h-[440px] flex-col gap-4 rounded-2xl border border-white/10 bg-white/[0.03] p-6 lg:col-span-3">
          <div className="flex items-center justify-between">
            <h2 className="font-mono text-xs font-bold uppercase tracking-[0.2em] text-slate-300">Conversation</h2>
            {total ? <span className="text-xs text-slate-500">Completed in {(total / 1000).toFixed(1)} s</span> : null}
          </div>

          {!result ? (
            <div className="flex flex-1 flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-white/10 text-center">
              <Bot className="h-9 w-9 text-violet-400/70" />
              <p className="font-semibold text-slate-200">Your conversation will appear here</p>
              <p className="max-w-xs text-sm text-slate-500">Record or upload a clip, then press Send to Companion.</p>
            </div>
          ) : (
            <div className="space-y-4">
              <article className="mr-8 space-y-3 rounded-xl border border-white/10 bg-white/[0.04] p-4">
                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-1.5 font-bold uppercase tracking-wider text-slate-400"><User className="h-3.5 w-3.5" /> You</span>
                  <span className="text-slate-500">{result.transcript_source === "manual" ? "Typed by you" : result.transcript ? "Google Speech-to-Text" : ""}</span>
                </div>
                <p className="text-[15px] leading-relaxed">{result.transcript || "We couldn't make out the words."}</p>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs font-semibold text-slate-500">Vocal emotion</span>
                  <EmotionChip cue={result.vocal_cue} />
                </div>
                <p className="text-[11px] text-slate-500">A tentative model observation, not a diagnosis.</p>
              </article>

              <article className="ml-8 space-y-3 rounded-xl border border-violet-500/25 bg-violet-500/10 p-4">
                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-1.5 font-bold uppercase tracking-wider text-violet-300"><Bot className="h-3.5 w-3.5" /> Companion</span>
                  <span className="text-slate-500">{result.resolved_model || ""}</span>
                </div>
                <p className="text-[15px] leading-relaxed">{result.reply_text || "No reply yet — please try again."}</p>
                {replyAudio && <audio controls src={replyAudio} className="w-full [color-scheme:dark]" />}
                <div className="flex flex-wrap gap-2">
                  <button type="button" onClick={speak} disabled={!result.reply_text || !!busy}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-black/30 px-3 py-1.5 text-sm font-medium transition hover:border-violet-400 hover:text-violet-200 disabled:opacity-40">
                    {busy === "speak" ? <Loader2 className="h-4 w-4 animate-spin" /> : <Volume2 className="h-4 w-4" />} Play reply
                  </button>
                  <button type="button" onClick={compare} disabled={!result.comparison_token || !result.vocal_cue || !!busy}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-black/30 px-3 py-1.5 text-sm font-medium transition hover:border-violet-400 hover:text-violet-200 disabled:opacity-40">
                    {busy === "compare" ? <Loader2 className="h-4 w-4 animate-spin" /> : <GitCompare className="h-4 w-4" />} Compare without emotion
                  </button>
                  {needsManual && (
                    <button type="button" onClick={send} disabled={!manual.trim() || !!busy}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-amber-400/30 bg-amber-400/10 px-3 py-1.5 text-sm font-medium text-amber-200 disabled:opacity-40">
                      <RotateCcw className="h-4 w-4" /> Send typed words
                    </button>
                  )}
                </div>
              </article>

              {notices.length > 0 && (
                <div className="flex gap-2 rounded-xl border border-amber-400/25 bg-amber-400/5 p-3 text-sm text-amber-200">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{notices.join(" ")}</span>
                </div>
              )}

              {comparison && (
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="rounded-xl border border-violet-500/25 bg-violet-500/5 p-4">
                    <p className="mb-2 font-mono text-[11px] font-bold uppercase tracking-widest text-violet-300">With vocal emotion</p>
                    <p className="text-sm leading-relaxed">{comparison.with}</p>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
                    <p className="mb-2 font-mono text-[11px] font-bold uppercase tracking-widest text-slate-400">Without vocal emotion</p>
                    <p className="text-sm leading-relaxed">{comparison.without}</p>
                  </div>
                  <p className="text-xs text-slate-500 sm:col-span-2">Same words, model ({comparison.model}) and settings — only the emotion cue removed.</p>
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
