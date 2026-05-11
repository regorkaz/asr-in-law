const API_BASE = window.location.origin;

const TARGET_SAMPLE_RATE = 16000;

let sessionId = null;

let enrollRecorder = null;
let enrollChunks = [];
let enrollTimer = null;
let enrollStartedAt = null;

let ws = null;
let audioContext = null;
let micStream = null;
let processor = null;
let sourceNode = null;
let sentSamples = 0;

const $ = (id) => document.getElementById(id);

const statusBadge = $("statusBadge");
const eventsLog = $("eventsLog");
const transcriptEl = $("transcript");

function setStatus(text, cls = "") {
  statusBadge.textContent = text;
  statusBadge.className = `badge ${cls}`.trim();
}

function logEvent(obj) {
  const text = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
  eventsLog.textContent = `${text}\n\n${eventsLog.textContent}`.slice(0, 30000);
}

function logError(err) {
  const message = err && err.stack ? err.stack : String(err);
  console.error(err);
  logEvent(message);
}

function setSessionControls(enabled) {
  $("startEnrollBtn").disabled = !enabled;
  $("enrollFile").disabled = !enabled;
  $("uploadEnrollBtn").disabled = !enabled;
  $("startStreamBtn").disabled = !enabled;

  const consultationFile = $("consultationFile");
  if (consultationFile) {
    consultationFile.disabled = !enabled;
  }

  const streamFileBtn = $("streamFileBtn");
  if (streamFileBtn) {
    streamFileBtn.disabled = !enabled;
  }

  const downloadBtn = $("downloadTranscriptBtn");
  if (downloadBtn) {
    downloadBtn.disabled = !enabled;
  }
}

function speakerClass(speaker) {
  if (speaker === "LAWYER") return "speaker-lawyer";
  if (speaker === "CLIENT") return "speaker-client";
  return "speaker-unknown";
}

function speakerText(speaker) {
  if (speaker === "LAWYER") return "Юрист";
  if (speaker === "CLIENT") return "Клиент";
  return "Неизвестно";
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function addSegment(segment) {
  const div = document.createElement("div");
  div.className = "segment";

  const speaker = segment.speaker || "UNKNOWN";
  const conf = segment.speaker_confidence;
  const sim = segment.speaker_similarity;

  const confText =
    conf === null || conf === undefined
      ? ""
      : ` conf=${Number(conf).toFixed(2)}`;

  const simText =
    sim === null || sim === undefined
      ? ""
      : ` sim=${Number(sim).toFixed(2)}`;

  const start = Number(segment.start_time || 0).toFixed(2);
  const end = Number(segment.end_time || 0).toFixed(2);

  div.innerHTML = `
    <div class="meta">
      <span>
        <span class="speaker ${speakerClass(speaker)}">${speakerText(speaker)}</span>
        ${confText}${simText}
      </span>
      <span>${start}–${end} сек</span>
    </div>
    <div class="text">${escapeHtml(segment.text || "")}</div>
  `;

  transcriptEl.appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
}

async function createSession() {
  const title = $("sessionTitle").value || null;
  const thresholdRaw = $("speakerThreshold").value;
  const threshold = thresholdRaw ? Number(thresholdRaw) : null;

  const resp = await fetch(`${API_BASE}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title,
      speaker_similarity_threshold: threshold,
    }),
  });

  if (!resp.ok) {
    throw new Error(await resp.text());
  }

  const data = await resp.json();
  sessionId = data.session_id;

  $("sessionId").textContent = data.session_id;

  setSessionControls(true);
  setStatus("Сессия создана", "ok");
  logEvent(data);
}

async function uploadEnrollFile(file) {
  if (!sessionId) {
    throw new Error("Session is not created.");
  }

  const form = new FormData();
  form.append("file", file, file.name || "lawyer_enrollment.webm");

  const resp = await fetch(`${API_BASE}/sessions/${sessionId}/enroll`, {
    method: "POST",
    body: form,
  });

  if (!resp.ok) {
    throw new Error(await resp.text());
  }

  const data = await resp.json();
  $("enrollResult").textContent = JSON.stringify(data, null, 2);
  logEvent(data);
}

function resetEnrollTimer() {
  enrollStartedAt = null;

  if (enrollTimer) {
    clearInterval(enrollTimer);
    enrollTimer = null;
  }

  const el = $("enrollAudioSec");
  if (el) {
    el.textContent = "0.0";
  }
}

function startEnrollTimer() {
  enrollStartedAt = performance.now();

  const el = $("enrollAudioSec");
  if (el) {
    el.textContent = "0.0";
  }

  if (enrollTimer) {
    clearInterval(enrollTimer);
  }

  enrollTimer = setInterval(() => {
    if (!enrollStartedAt) {
      return;
    }

    const elapsed = (performance.now() - enrollStartedAt) / 1000.0;

    if (el) {
      el.textContent = elapsed.toFixed(1);
    }
  }, 100);
}

function stopEnrollTimer() {
  if (enrollTimer) {
    clearInterval(enrollTimer);
    enrollTimer = null;
  }

  if (enrollStartedAt) {
    const elapsed = (performance.now() - enrollStartedAt) / 1000.0;
    const el = $("enrollAudioSec");

    if (el) {
      el.textContent = elapsed.toFixed(1);
    }
  }
}

async function startEnrollRecording() {
  if (!sessionId) {
    throw new Error("Session is not created.");
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("getUserMedia is not available. Use localhost or HTTPS.");
  }

  if (typeof MediaRecorder === "undefined") {
    throw new Error("MediaRecorder is not available in this browser.");
  }

  enrollChunks = [];
  resetEnrollTimer();

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  let options = {};

  if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
    options = { mimeType: "audio/webm;codecs=opus" };
  } else if (MediaRecorder.isTypeSupported("audio/webm")) {
    options = { mimeType: "audio/webm" };
  }

  enrollRecorder = new MediaRecorder(stream, options);

  enrollRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      enrollChunks.push(event.data);
    }
  };

  enrollRecorder.onerror = (event) => {
    logError(event.error || event);
    setStatus("Ошибка MediaRecorder", "err");
  };

  enrollRecorder.onstart = () => {
    setStatus("Запись голоса юриста", "warn");
    startEnrollTimer();
    logEvent("Enrollment recording started");
  };

  enrollRecorder.onstop = async () => {
    stopEnrollTimer();

    try {
      const blob = new Blob(enrollChunks, {
        type: enrollRecorder.mimeType || "audio/webm",
      });

      if (blob.size === 0) {
        throw new Error("Enrollment recording is empty.");
      }

      const file = new File([blob], "lawyer_enrollment.webm", {
        type: blob.type,
      });

      await uploadEnrollFile(file);
      setStatus("Голос юриста записан", "ok");
    } catch (err) {
      setStatus("Ошибка enrollment", "err");
      logError(err);
    } finally {
      stream.getTracks().forEach((track) => track.stop());
      $("startEnrollBtn").disabled = false;
      $("stopEnrollBtn").disabled = true;
    }
  };

  enrollRecorder.start(250);

  $("startEnrollBtn").disabled = true;
  $("stopEnrollBtn").disabled = false;
}

function stopEnrollRecording() {
  if (enrollRecorder && enrollRecorder.state !== "inactive") {
    enrollRecorder.stop();
  }
}

function wsUrl() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws/audio/${sessionId}`;
}

function openAudioWebSocket() {
  return new Promise((resolve, reject) => {
    ws = new WebSocket(wsUrl());
    ws.binaryType = "arraybuffer";

    let settled = false;

    ws.onopen = () => {
      $("wsState").textContent = "connected";
      setStatus("WebSocket подключен", "ok");
      logEvent("WebSocket connected");

      settled = true;
      resolve(ws);
    };

    ws.onmessage = (event) => {
      try {
        const obj = JSON.parse(event.data);
        logEvent(obj);

        if (obj.event === "partial") {
          const segments = obj.payload?.segments || [];
          for (const seg of segments) {
            addSegment(seg);
          }
        }

        if (obj.event === "final") {
          $("wsState").textContent = "finalized";
          setStatus("Финализировано", "ok");

          const segments = obj.payload?.segments || [];
          transcriptEl.innerHTML = "";

          for (const seg of segments) {
            addSegment(seg);
          }

          const downloadBtn = $("downloadTranscriptBtn");
          if (downloadBtn) {
            downloadBtn.disabled = false;
          }
        }

        if (obj.event === "error") {
          setStatus("Ошибка сервера", "err");
        }
      } catch (err) {
        logError(err);
        logEvent(event.data);
      }
    };

    ws.onerror = () => {
      $("wsState").textContent = "error";
      setStatus("WebSocket error", "err");
      logEvent("WebSocket error");

      if (!settled) {
        settled = true;
        reject(new Error("WebSocket error"));
      }
    };

    ws.onclose = () => {
      $("wsState").textContent = "closed";
      stopMicAudioOnly();

      if (!settled) {
        settled = true;
        reject(new Error("WebSocket closed before connection was established."));
      }
    };
  });
}

async function startStreamingMic() {
  if (!sessionId) {
    throw new Error("Session is not created.");
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("getUserMedia is not available. Use localhost or HTTPS.");
  }

  sentSamples = 0;
  $("sentAudioSec").textContent = "0.0";
  $("audioSourceMode").textContent = "microphone";

  $("startStreamBtn").disabled = true;
  $("stopStreamBtn").disabled = false;

  const streamFileBtn = $("streamFileBtn");
  if (streamFileBtn) {
    streamFileBtn.disabled = true;
  }

  setStatus("Запуск микрофона", "warn");


  await startMicAudio();
  await openAudioWebSocket();

  setStatus("Микрофон подключен", "ok");
}

async function startMicAudio() {
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  const AudioContextClass = window.AudioContext || window.webkitAudioContext;

  if (!AudioContextClass) {
    throw new Error("AudioContext is not available in this browser.");
  }

  audioContext = new AudioContextClass();

  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }

  logEvent({
    message: "AudioContext started",
    sampleRate: audioContext.sampleRate,
    state: audioContext.state,
  });

  sourceNode = audioContext.createMediaStreamSource(micStream);

  const bufferSize = 4096;
  processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

  processor.onaudioprocess = (event) => {
    try {
      const input = event.inputBuffer.getChannelData(0);

      if (!input || input.length === 0) {
        return;
      }

      const resampled = resampleLinear(
        input,
        audioContext.sampleRate,
        TARGET_SAMPLE_RATE,
      );

      if (!resampled || resampled.length === 0) {
        return;
      }

      const pcm16 = floatTo16BitPCM(resampled);

      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(pcm16.buffer);

        sentSamples += resampled.length;
        $("sentAudioSec").textContent = (sentSamples / TARGET_SAMPLE_RATE).toFixed(1);
      }
    } catch (err) {
      logError(err);
    }
  };

  sourceNode.connect(processor);

  processor.connect(audioContext.destination);
}

function stopStreamingMic() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "finalize" }));
  }

  stopMicAudioOnly();
}

function stopMicAudioOnly() {
  if (processor) {
    try {
      processor.disconnect();
    } catch {}
    processor.onaudioprocess = null;
    processor = null;
  }

  if (sourceNode) {
    try {
      sourceNode.disconnect();
    } catch {}
    sourceNode = null;
  }

  if (audioContext) {
    try {
      audioContext.close();
    } catch {}
    audioContext = null;
  }

  if (micStream) {
    micStream.getTracks().forEach((track) => track.stop());
    micStream = null;
  }

  $("startStreamBtn").disabled = !sessionId;
  $("stopStreamBtn").disabled = true;

  const streamFileBtn = $("streamFileBtn");
  if (streamFileBtn) {
    streamFileBtn.disabled = !sessionId;
  }
}

async function streamConsultationFile() {
  if (!sessionId) {
    throw new Error("Session is not created.");
  }

  const input = $("consultationFile");
  const file = input?.files?.[0];

  if (!file) {
    throw new Error("Choose consultation audio file first.");
  }

  sentSamples = 0;
  $("sentAudioSec").textContent = "0.0";
  $("audioSourceMode").textContent = "file";

  $("startStreamBtn").disabled = true;
  $("stopStreamBtn").disabled = false;

  const streamFileBtn = $("streamFileBtn");
  if (streamFileBtn) {
    streamFileBtn.disabled = true;
  }

  setStatus("Чтение аудиофайла", "warn");

  const audio = await decodeAudioFileToMono16k(file);

  logEvent({
    message: "Consultation file decoded",
    file: file.name,
    samples: audio.length,
    sampleRate: TARGET_SAMPLE_RATE,
    durationSec: audio.length / TARGET_SAMPLE_RATE,
  });

  await openAudioWebSocket();

  setStatus("Отправка файла как stream", "warn");

  await sendAudioFloat32Realtime(audio, TARGET_SAMPLE_RATE, 300);

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "finalize" }));
  }

  setStatus("Файл отправлен, ожидание финализации", "warn");
}

async function decodeAudioFileToMono16k(file) {
  const arrayBuffer = await file.arrayBuffer();

  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error("AudioContext is not available in this browser.");
  }

  const ctx = new AudioContextClass();

  try {
    const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));

    const mono = mixToMono(decoded);
    const resampled = resampleLinear(mono, decoded.sampleRate, TARGET_SAMPLE_RATE);

    return resampled;
  } finally {
    try {
      await ctx.close();
    } catch {}
  }
}

function mixToMono(audioBuffer) {
  const channels = audioBuffer.numberOfChannels;
  const length = audioBuffer.length;

  if (channels === 1) {
    return new Float32Array(audioBuffer.getChannelData(0));
  }

  const mono = new Float32Array(length);

  for (let ch = 0; ch < channels; ch++) {
    const data = audioBuffer.getChannelData(ch);

    for (let i = 0; i < length; i++) {
      mono[i] += data[i] / channels;
    }
  }

  return mono;
}

async function sendAudioFloat32Realtime(audio, sampleRate, blockMs) {
  const blockSamples = Math.max(1, Math.floor(sampleRate * blockMs / 1000));
  const totalSamples = audio.length;

  for (let start = 0; start < totalSamples; start += blockSamples) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      throw new Error("WebSocket is not open while streaming file.");
    }

    const end = Math.min(start + blockSamples, totalSamples);
    const chunk = audio.subarray(start, end);
    const pcm16 = floatTo16BitPCM(chunk);

    ws.send(pcm16.buffer);

    sentSamples += chunk.length;
    $("sentAudioSec").textContent = (sentSamples / sampleRate).toFixed(1);

    await sleep(blockMs);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function loadTranscript() {
  if (!sessionId) {
    return null;
  }

  const resp = await fetch(`${API_BASE}/sessions/${sessionId}/transcript`);

  if (!resp.ok) {
    throw new Error(await resp.text());
  }

  const data = await resp.json();
  transcriptEl.innerHTML = "";

  for (const seg of data.segments || []) {
    addSegment(seg);
  }

  logEvent(data);
  return data;
}

async function downloadTranscriptJson() {
  if (!sessionId) {
    throw new Error("Session is not created.");
  }

  const resp = await fetch(`${API_BASE}/sessions/${sessionId}/transcript`);

  if (!resp.ok) {
    throw new Error(await resp.text());
  }

  const data = await resp.json();
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json;charset=utf-8",
  });

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");

  a.href = url;
  a.download = `transcript_${sessionId}.json`;
  document.body.appendChild(a);
  a.click();

  a.remove();
  URL.revokeObjectURL(url);
}

function floatTo16BitPCM(float32Array) {
  const out = new Int16Array(float32Array.length);

  for (let i = 0; i < float32Array.length; i++) {
    const sample = Math.max(-1, Math.min(1, float32Array[i]));
    out[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }

  return out;
}

function resampleLinear(input, inputRate, outputRate) {
  if (inputRate === outputRate) {
    return new Float32Array(input);
  }

  const ratio = inputRate / outputRate;
  const outputLength = Math.floor(input.length / ratio);
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;

    const s0 = input[idx] || 0;
    const s1 = input[idx + 1] || s0;

    output[i] = s0 + frac * (s1 - s0);
  }

  return output;
}

$("createSessionBtn").addEventListener("click", async () => {
  try {
    await createSession();
  } catch (err) {
    setStatus("Ошибка создания сессии", "err");
    logError(err);
  }
});

$("startEnrollBtn").addEventListener("click", async () => {
  try {
    await startEnrollRecording();
  } catch (err) {
    setStatus("Ошибка записи голоса юриста", "err");
    logError(err);
    $("startEnrollBtn").disabled = false;
    $("stopEnrollBtn").disabled = true;
  }
});

$("stopEnrollBtn").addEventListener("click", stopEnrollRecording);

$("uploadEnrollBtn").addEventListener("click", async () => {
  try {
    const file = $("enrollFile").files[0];

    if (!file) {
      throw new Error("Choose enrollment audio file first.");
    }

    await uploadEnrollFile(file);
    setStatus("Enrollment загружен", "ok");
  } catch (err) {
    setStatus("Ошибка enrollment", "err");
    logError(err);
  }
});

$("startStreamBtn").addEventListener("click", async () => {
  try {
    await startStreamingMic();
  } catch (err) {
    setStatus("Ошибка микрофона", "err");
    logError(err);
    stopMicAudioOnly();
  }
});

$("stopStreamBtn").addEventListener("click", stopStreamingMic);

const streamFileBtn = $("streamFileBtn");
if (streamFileBtn) {
  streamFileBtn.addEventListener("click", async () => {
    try {
      await streamConsultationFile();
    } catch (err) {
      setStatus("Ошибка отправки файла", "err");
      logError(err);
      stopMicAudioOnly();

      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.close();
        } catch {}
      }
    }
  });
}

const downloadTranscriptBtn = $("downloadTranscriptBtn");
if (downloadTranscriptBtn) {
  downloadTranscriptBtn.addEventListener("click", async () => {
    try {
      await downloadTranscriptJson();
    } catch (err) {
      setStatus("Ошибка скачивания transcript", "err");
      logError(err);
    }
  });
}

setStatus("Ожидание сессии", "warn");
setSessionControls(false);
resetEnrollTimer();