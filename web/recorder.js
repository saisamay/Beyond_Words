const $ = selector => document.querySelector(selector);
const consent = $("#consent");
const record = $("#record");
const stop = $("#stop");
const status = $("#status");
const playback = $("#playback");
const download = $("#download");
const send = $("#send");
const clip = $("#clip");
const manualPanel = $("#manual-panel");
const manual = $("#manual");
const results = $("#results");
const compare = $("#compare");
const speak = $("#speak");
const replyAudio = $("#reply-audio");
const supportDemo = $("#support-demo");
const recorderUi = $("#recorder-ui");
const meter = $("#meter");
const timer = $("#timer");
const clipPreview = $("#clip-preview");
const emptyState = $("#empty-state");
const dropzone = $("#dropzone");
const MAX_SECONDS = 12;

let replyText;
let speechUrl;
let recordingBlob;
let comparisonToken;
let busy = false;
let recordingUrl;
let recorder;
let stopTimer;
let meterFrame;
let audioContext;
let recordStarted;

/* ---------- Header health ---------- */
function setPill(pill, state, text) {
  pill.classList.remove("ok", "warn", "bad");
  if (state) pill.classList.add(state);
  pill.lastElementChild.textContent = text;
}

async function refreshHealth() {
  try {
    const health = await fetch("/health", { signal: AbortSignal.timeout(5000) });
    setPill($("#server-pill"), health.ok ? "ok" : "bad", health.ok ? "Server online" : "Server error");
    const info = await (await fetch("/v1/status", { signal: AbortSignal.timeout(5000) })).json();
    const loaded = info.vocal_model?.loaded;
    setPill($("#model-pill"), loaded ? "ok" : "warn", loaded ? "Emotion model ready" : "Emotion model off");
  } catch {
    setPill($("#server-pill"), "bad", "Server offline");
    setPill($("#model-pill"), null, "Emotion model");
  }
}
refreshHealth();

/* ---------- Level meter ---------- */
function meterColors() {
  const styles = getComputedStyle(document.documentElement);
  return { bar: styles.getPropertyValue("--brand").trim() || "#4f46e5", idle: styles.getPropertyValue("--border").trim() || "#e2e7f0" };
}

function drawIdleMeter() {
  const context = meter.getContext("2d");
  const { idle } = meterColors();
  context.clearRect(0, 0, meter.width, meter.height);
  const bars = 48;
  const gap = meter.width / bars;
  context.fillStyle = idle;
  for (let i = 0; i < bars; i++) {
    context.fillRect(i * gap + gap * 0.3, meter.height / 2 - 2, gap * 0.4, 4);
  }
}

function startMeter(stream) {
  try {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    audioContext.createMediaStreamSource(stream).connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    const context = meter.getContext("2d");
    const { bar } = meterColors();
    const bars = 48;
    const gap = meter.width / bars;
    const draw = () => {
      analyser.getByteFrequencyData(data);
      context.clearRect(0, 0, meter.width, meter.height);
      context.fillStyle = bar;
      for (let i = 0; i < bars; i++) {
        const value = data[Math.floor((i / bars) * data.length * 0.7)] / 255;
        const height = Math.max(4, value * meter.height * 0.95);
        context.fillRect(i * gap + gap * 0.3, (meter.height - height) / 2, gap * 0.4, height);
      }
      const elapsed = Math.min(MAX_SECONDS, (performance.now() - recordStarted) / 1000);
      timer.firstChild.textContent = `0:${String(Math.floor(elapsed)).padStart(2, "0")} `;
      meterFrame = requestAnimationFrame(draw);
    };
    draw();
  } catch {
    /* The meter is decorative; recording continues without it. */
  }
}

function stopMeter() {
  cancelAnimationFrame(meterFrame);
  audioContext?.close().catch(() => {});
  audioContext = undefined;
  recorderUi.dataset.state = "idle";
  drawIdleMeter();
}
drawIdleMeter();

/* ---------- Controls ---------- */
function resetControls(message) {
  clearTimeout(stopTimer);
  stopMeter();
  recorder = undefined;
  consent.disabled = false;
  clip.disabled = false;
  record.disabled = !consent.checked;
  stop.disabled = true;
  send.disabled = !recordingBlob || !consent.checked || busy;
  compare.disabled = !comparisonToken;
  speak.disabled = !replyText;
  supportDemo.disabled = false;
  status.textContent = message;
}

function showRecording(blob) {
  clearReplyAudio();
  replyText = undefined;
  speak.disabled = true;
  recordingBlob = blob;
  comparisonToken = undefined;
  results.hidden = true;
  emptyState.hidden = false;
  $("#turn-meta").textContent = "";
  manualPanel.hidden = true;
  manual.value = "";
  $("#comparison").replaceChildren();
  if (recordingUrl) URL.revokeObjectURL(recordingUrl);
  recordingUrl = URL.createObjectURL(blob);
  const extension = {
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/mp4": "m4a",
    "audio/wav": "wav",
  }[blob.type.split(";")[0]] ?? "audio";
  playback.src = recordingUrl;
  playback.hidden = false;
  clipPreview.hidden = false;
  download.href = recordingUrl;
  download.download = `beyond-words-sample.${extension}`;
  download.hidden = false;
  send.disabled = !consent.checked;
}

consent.addEventListener("change", () => {
  record.disabled = !consent.checked;
  send.disabled = !consent.checked || !recordingBlob || busy;
  status.textContent = consent.checked ? "Ready — tap the microphone to record." : "Give microphone consent to begin.";
});

record.addEventListener("click", async () => {
  record.disabled = true;
  consent.disabled = true;
  send.disabled = true;
  clip.disabled = true;
  compare.disabled = true;
  speak.disabled = true;
  supportDemo.disabled = true;
  status.textContent = "Requesting microphone access…";
  let stream;
  try {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      resetControls("Open this page in a supported browser on localhost or HTTPS.");
      return;
    }
    // Ask the browser for a cleaner, level-adjusted mono voice signal; browsers ignore
    // constraints they cannot honor, so this never blocks recording.
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    const chunks = [];
    recorder = new MediaRecorder(stream, { audioBitsPerSecond: 128000 });
    let recordingFailed = false;
    recorder.addEventListener("dataavailable", event => {
      if (event.data.size) chunks.push(event.data);
    });
    recorder.addEventListener("error", () => {
      recordingFailed = true;
    });
    recorder.addEventListener("stop", () => {
      stream.getTracks().forEach(track => track.stop());
      if (recordingFailed || !chunks.length) {
        resetControls("No audio was captured. Try recording again.");
        return;
      }
      showRecording(new Blob(chunks, { type: chunks[0].type || recorder.mimeType }));
      resetControls("Recording ready. Play it back, then Send to Companion.");
    }, { once: true });
    recorder.start();
    recordStarted = performance.now();
    recorderUi.dataset.state = "recording";
    startMeter(stream);
    stop.disabled = false;
    status.textContent = `Listening… speak naturally (up to ${MAX_SECONDS} seconds).`;
    stopTimer = setTimeout(() => stop.click(), MAX_SECONDS * 1000);
  } catch {
    stream?.getTracks().forEach(track => track.stop());
    resetControls("Could not start recording. Check microphone permissions and try again.");
  }
});

stop.addEventListener("click", () => {
  stop.disabled = true;
  if (recorder?.state === "recording") recorder.stop();
});

clip.addEventListener("change", () => {
  dropzone.classList.remove("over");
  const file = clip.files[0];
  if (!file) return;
  if (!file.size || file.size > 8 * 1024 * 1024) {
    status.textContent = "Choose a non-empty clip smaller than 8 MiB.";
    return;
  }
  showRecording(file);
  status.textContent = `“${file.name}” ready. Play it back, then Send to Companion.`;
});
["dragenter", "dragover"].forEach(type => dropzone.addEventListener(type, () => dropzone.classList.add("over")));
["dragleave", "drop"].forEach(type => dropzone.addEventListener(type, () => dropzone.classList.remove("over")));

function setBusy(value, message) {
  busy = value;
  document.body.classList.toggle("is-busy", value);
  consent.disabled = value;
  clip.disabled = value;
  record.disabled = value || !consent.checked;
  send.disabled = value || !consent.checked || !recordingBlob;
  compare.disabled = value || !comparisonToken;
  speak.disabled = value || !replyText;
  supportDemo.disabled = value;
  manual.disabled = value;
  if (message) status.textContent = message;
}

async function post(url, options) {
  const response = await fetch(url, { method: "POST", ...options, signal: AbortSignal.timeout(120000) });
  const data = await response.json();
  if (!response.ok && !Object.hasOwn(data, "transcript")) {
    throw new Error(typeof data.detail === "string" ? data.detail : "Please provide a valid recording and try again.");
  }
  return data;
}

/* ---------- Companion turn ---------- */
send.addEventListener("click", async () => {
  if (!recordingBlob || busy || !consent.checked) return;
  setBusy(true, "Transcribing your words and preparing a reply… the first turn may take longer.");
  results.hidden = true;
  clearReplyAudio();
  replyText = undefined;
  comparisonToken = undefined;
  $("#comparison").replaceChildren();
  try {
    const form = new FormData();
    form.append("audio", recordingBlob, download.download);
    if (!manualPanel.hidden && manual.value.trim()) form.append("manual_transcript", manual.value.trim());
    const data = await post("/v1/turn", { body: form });
    emptyState.hidden = true;
    results.hidden = false;
    $("#transcript").textContent = data.transcript || "Transcription unavailable.";
    $("#transcript-source").textContent = data.transcript_source === "manual"
      ? "Manual transcription — supplied by you" : data.transcript ? "Google Speech-to-Text" : "";

    const cue = $("#vocal-cue");
    cue.textContent = data.vocal_cue
      ? `${data.vocal_cue.label} · ${data.vocal_cue.model_version}` : "Unavailable — using your words alone.";
    cue.dataset.label = data.vocal_cue ? String(data.vocal_cue.label).toLowerCase() : "unavailable";

    $("#reply").textContent = data.reply_text || "No reply yet. Please retry.";
    $("#reply-model").textContent = data.resolved_model || "";
    const total = data.timings_ms?.total_ms;
    $("#turn-meta").textContent = total ? `Turn completed in ${(total / 1000).toFixed(1)} s` : "";
    replyText = data.reply_text;

    $("#fallback").textContent = (data.fallback_status || []).map(value => ({
      manual_transcription: "Using your manual transcription.",
      vocal_cue_unavailable: "Vocal cue unavailable; transcript-only reply.",
      manual_transcript_required: "We couldn't make out the words. Type what you said and retry with the same clip.",
      reply_unavailable: "Google reply unavailable. Your clip remains ready to retry.",
      gemini_fallback: "Primary model busy — answered by the fallback model.",
    }[value] || value)).join(" ");
    manualPanel.hidden = !data.fallback_status.includes("manual_transcript_required") && data.transcript_source !== "manual";
    comparisonToken = data.reply_text && data.vocal_cue ? data.comparison_token : undefined;
    status.textContent = data.error || "Companion turn ready.";
    if (!manualPanel.hidden) manual.focus();
  } catch (error) {
    status.textContent = `${error.name === "TimeoutError" ? "Request timed out." : error.message} Your clip is kept here. Retry Send.`;
  } finally {
    setBusy(false);
  }
});

/* ---------- Comparison ---------- */
compare.addEventListener("click", async () => {
  if (!comparisonToken || busy) return;
  setBusy(true, "Comparing the same transcript and prompt without the vocal cue…");
  try {
    const data = await post("/v1/compare", {
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comparison_token: comparisonToken }),
    });
    if (data.replace_original_reply && data.cue_on_reply_text) {
      $("#reply").textContent = data.cue_on_reply_text;
      replyText = data.cue_on_reply_text;
      $("#reply-model").textContent = data.cue_on_model || "";
    }
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = "Without vocal cue:";
    const text = document.createElement("p");
    text.textContent = data.cue_off_reply_text ?? data.reply_text;
    const note = document.createElement("small");
    note.className = "hint";
    note.textContent = `Same transcript, model (${data.resolved_model || "Gemini"}), prompt and settings — only the vocal cue removed.`;
    $("#comparison").replaceChildren(label, text, note);
    status.textContent = "Comparison ready.";
  } catch (error) {
    status.textContent = `${error.message} You can retry Compare.`;
  } finally {
    setBusy(false);
  }
});

/* ---------- Speech playback ---------- */
function clearReplyAudio() {
  replyAudio.pause();
  replyAudio.hidden = true;
  replyAudio.removeAttribute("src");
  if (speechUrl) URL.revokeObjectURL(speechUrl);
  speechUrl = undefined;
}

speak.addEventListener("click", async () => {
  if (!replyText || busy) return;
  setBusy(true, "Preparing Google speech playback…");
  try {
    const response = await fetch("/v1/speak", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reply_text: replyText }), signal: AbortSignal.timeout(35000),
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Speech playback unavailable.");
    }
    clearReplyAudio();
    speechUrl = URL.createObjectURL(await response.blob());
    replyAudio.src = speechUrl;
    replyAudio.hidden = false;
    await replyAudio.play();
    status.textContent = "Playing Companion reply.";
  } catch (error) {
    status.textContent = `${error.message} Your text reply is kept. Retry Play Companion reply.`;
  } finally {
    setBusy(false);
  }
});

/* ---------- Support scenario ---------- */
supportDemo.addEventListener("click", async () => {
  setBusy(true, "Preparing the fictional support example…");
  try {
    const data = await post("/v1/support-demo", {});
    $("#support-result").hidden = false;
    $("#support-transcript").textContent = data.transcript;
    $("#support-reply").textContent = data.reply_text || data.error;
    const brief = $("#handover");
    brief.replaceChildren();
    for (const [key, value] of Object.entries(data.handover_brief)) {
      const term = document.createElement("dt");
      term.textContent = key.replaceAll("_", " ");
      const description = document.createElement("dd");
      description.textContent = value;
      brief.append(term, description);
    }
    status.textContent = data.error || "Fictional support example ready. The handover brief has not been sent.";
  } catch (error) {
    status.textContent = `${error.message} Retry the support example.`;
  } finally {
    setBusy(false);
  }
});
