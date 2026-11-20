const consent = document.querySelector("#consent");
const record = document.querySelector("#record");
const stop = document.querySelector("#stop");
const status = document.querySelector("#status");
const playback = document.querySelector("#playback");
const download = document.querySelector("#download");

let recordingUrl;
let recorder;
let stopTimer;

function resetControls(message) {
  clearTimeout(stopTimer);
  recorder = undefined;
  consent.disabled = false;
  record.disabled = !consent.checked;
  stop.disabled = true;
  status.textContent = message;
}

function showRecording(blob) {
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
}

consent.addEventListener("change", () => {
  record.disabled = !consent.checked;
  status.textContent = consent.checked ? "Ready to record." : "Give microphone consent to begin.";
});

record.addEventListener("click", async () => {
  record.disabled = true;
  consent.disabled = true;
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
      resetControls("Recording ready. Play it back or download it.");
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
