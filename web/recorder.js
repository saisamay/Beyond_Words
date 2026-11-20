const consent = document.querySelector("#consent");
const record = document.querySelector("#record");
const stop = document.querySelector("#stop");
const status = document.querySelector("#status");
const playback = document.querySelector("#playback");
const download = document.querySelector("#download");
const send = document.querySelector("#send");
const clip = document.querySelector("#clip");
const manualPanel = document.querySelector("#manual-panel");
const manual = document.querySelector("#manual");
const results = document.querySelector("#results");
const compare = document.querySelector("#compare");
const speak = document.querySelector("#speak");
const replyAudio = document.querySelector("#reply-audio");
const supportDemo = document.querySelector("#support-demo");
let replyText;
let speechUrl;
let recordingBlob;
let comparisonToken;
let busy = false;

let recordingUrl;
let recorder;
let stopTimer;

function resetControls(message) {
  clearTimeout(stopTimer);
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
  manualPanel.hidden = true;
  manual.value = "";
  document.querySelector("#comparison").textContent = "";
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
  download.href = recordingUrl;
  download.download = `beyond-words-sample.${extension}`;
  download.hidden = false;
  send.disabled = !consent.checked;
}

consent.addEventListener("change", () => {
  record.disabled = !consent.checked;
  send.disabled = !consent.checked || !recordingBlob || busy;
  status.textContent = consent.checked ? "Ready to record." : "Give microphone consent to begin.";
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
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const chunks = [];
    recorder = new MediaRecorder(stream);
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
    stop.disabled = false;
    status.textContent = "Recording for up to 12 seconds…";
    stopTimer = setTimeout(() => stop.click(), 12000);
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
  const file = clip.files[0];
  if (!file) return;
  if (!file.size || file.size > 8 * 1024 * 1024) {
    status.textContent = "Choose a non-empty clip smaller than 8 MiB.";
    return;
  }
  showRecording(file);
  status.textContent = "Clip ready. Play it back, then Send to Companion.";
});

function setBusy(value, message) {
  busy = value;
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

send.addEventListener("click", async () => {
  if (!recordingBlob || busy || !consent.checked) return;
  setBusy(true, "Processing… transcribing your words and preparing a reply. The first turn may take longer.");
  results.hidden = true;
  clearReplyAudio();
  replyText = undefined;
  comparisonToken = undefined;
  document.querySelector("#comparison").textContent = "";
  try {
    const form = new FormData();
    form.append("audio", recordingBlob, download.download);
    if (!manualPanel.hidden && manual.value.trim()) form.append("manual_transcript", manual.value.trim());
    const data = await post("/v1/turn", { body: form });
    results.hidden = false;
    document.querySelector("#transcript").textContent = data.transcript || "Transcription unavailable.";
    document.querySelector("#transcript-source").textContent = data.transcript_source === "manual"
      ? "Manual transcription — supplied by you" : data.transcript ? "Google Speech-to-Text" : "";
    document.querySelector("#vocal-cue").textContent = data.vocal_cue
      ? `${data.vocal_cue.label} · ${data.vocal_cue.model_version}` : "Unavailable — using your words alone.";
    document.querySelector("#reply").textContent = data.reply_text || "No reply yet. Please retry.";
    replyText = data.reply_text;
    document.querySelector("#fallback").textContent = (data.fallback_status || []).map(value => ({
      manual_transcription: "Using your manual transcription.",
      vocal_cue_unavailable: "Vocal cue unavailable; transcript-only reply.",
      manual_transcript_required: "Enter what you said below and retry with the same clip.",
      reply_unavailable: "Google reply unavailable. Your clip remains ready to retry.",
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

compare.addEventListener("click", async () => {
  if (!comparisonToken || busy) return;
  setBusy(true, "Comparing the same transcript and prompt without the vocal cue…");
  try {
    const data = await post("/v1/compare", {
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comparison_token: comparisonToken }),
    });
    document.querySelector("#comparison").textContent = `Without vocal cue: ${data.reply_text}`;
    status.textContent = "Comparison ready. Same transcript, Gemini model, prompt, and settings; vocal cue omitted.";
  } catch (error) {
    status.textContent = `${error.message} You can retry Compare.`;
  } finally {
    setBusy(false);
  }
});

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

supportDemo.addEventListener("click", async () => {
  setBusy(true, "Preparing the fictional support example…");
  try {
    const data = await post("/v1/support-demo", {});
    document.querySelector("#support-result").hidden = false;
    document.querySelector("#support-transcript").textContent = data.transcript;
    document.querySelector("#support-reply").textContent = data.reply_text || data.error;
    const brief = document.querySelector("#handover");
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
