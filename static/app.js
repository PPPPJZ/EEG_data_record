"use strict";

const CHANNEL_COUNT = 8;
const SAMPLE_RATE = 250;
const PREVIEW_SECONDS = 5;
const PREVIEW_POINTS = SAMPLE_RATE * PREVIEW_SECONDS;
const DISPLAY_LIMIT = 50;
const TRIAL_SECONDS = 8;
const BREAK_SECONDS = 33;
const ACTIVE_STAGES = new Set([
  "preparing",
  "rest",
  "cue",
  "imagery",
  "self_learning",
  "break_instruction",
  "break_monitor",
]);
const COLORS = [
  "#006d77",
  "#d1495b",
  "#3a7d44",
  "#7b2cbf",
  "#e07a1f",
  "#1d4e89",
  "#9c6644",
  "#495057",
];

const modeView = document.getElementById("modeView");
const informationView = document.getElementById("informationView");
const deviceView = document.getElementById("deviceView");
const experimentView = document.getElementById("experimentView");
const completeView = document.getElementById("completeView");
const sessionForm = document.getElementById("sessionForm");
const formError = document.getElementById("formError");
const confirmButton = document.getElementById("confirmButton");
const startButton = document.getElementById("startButton");
const breakToggleButton = document.getElementById("breakToggleButton");
const editSessionButton = document.getElementById("editSessionButton");
const newSessionButton = document.getElementById("newSessionButton");
const completeNewButton = document.getElementById("completeNewButton");
const deviceStatus = document.getElementById("deviceStatus");
const deviceStatusText = document.getElementById("deviceStatusText");
const deviceMessage = document.getElementById("deviceMessage");
const deviceMessageText = document.getElementById("deviceMessageText");
const packetStatus = document.getElementById("packetStatus");
const waveformCanvas = document.getElementById("waveformCanvas");
const deviceKicker = document.getElementById("deviceKicker");
const deviceTitle = document.getElementById("deviceTitle");
const breakTimer = document.getElementById("breakTimer");
const breakTimerStatus = document.getElementById("breakTimerStatus");
const breakCountdown = document.getElementById("breakCountdown");
const preparationContent = document.getElementById("preparationContent");
const preparationCountdown = document.getElementById("preparationCountdown");
const restContent = document.getElementById("restContent");
const remainingTrials = document.getElementById("remainingTrials");
const cueContent = document.getElementById("cueContent");
const directionArrow = document.getElementById("directionArrow");
const breakInstructionContent = document.getElementById(
  "breakInstructionContent",
);
const experimentProgress = document.getElementById("experimentProgress");
const experimentProgressFill = document.getElementById(
  "experimentProgressFill",
);
const hapticIntensity = document.getElementById("hapticIntensity");
const hapticTestButton = document.getElementById("hapticTestButton");
const hapticPanel = document.getElementById("hapticPanel");
const hapticMessage = document.getElementById("hapticMessage");

const previewData = Array.from({ length: CHANNEL_COUNT }, () => []);
let socket = null;
let latestState = null;
let previewNeedsDraw = true;
let audioContext = null;
let lastChimeSerial = null;
let selectedMode = null;
let hapticRequestError = "";
let modelRequestError = "";
let feedbackReceivedAt = 0;

function localDateString() {
  const now = new Date();
  const offsetMilliseconds = now.getTimezoneOffset() * 60 * 1000;
  return new Date(now.getTime() - offsetMilliseconds)
    .toISOString()
    .slice(0, 10);
}

document.getElementById("experimentDate").value = localDateString();

function showOnly(view) {
  for (const candidate of [
    modeView,
    informationView,
    deviceView,
    experimentView,
    completeView,
  ]) {
    candidate.hidden = candidate !== view;
  }
  document.body.classList.toggle("experiment-mode", view === experimentView);
  if (view === deviceView) {
    previewNeedsDraw = true;
  }
}

function updateHapticIntensity() {
  const high = Number(hapticIntensity.value);
  document.getElementById("hapticIntensityValue").value = high.toFixed(2);
  document.getElementById("hapticLowValue").value =
    Math.max(0, high - 0.15).toFixed(2);
}

function chooseMode(mode) {
  selectedMode = mode;
  document.getElementById("selectedMode").textContent = mode;
  document.getElementById("hapticIntensityField").hidden = mode !== "tactile_MI";
  updateModeFields();
  updateTrialSummary();
  updateHapticIntensity();
  showOnly(informationView);
}

document.getElementById("pureModeButton")
  .addEventListener("click", () => chooseMode("pure_MI"));
document.getElementById("tactileModeButton")
  .addEventListener("click", () => chooseMode("tactile_MI"));
document.getElementById("selfLearningModeButton")
  .addEventListener("click", () => chooseMode("self_learning"));

function updateModeFields() {
  const learning = selectedMode === "self_learning";
  for (const name of ["leftTrials", "rightTrials"]) {
    document.getElementById(`${name}Field`).hidden = learning;
    document.getElementById(name).disabled = learning;
  }
  for (const [field, input] of [
    ["learningTrialsField", "learningTrials"], ["teacherPathField", "teacherModelPath"],
  ]) {
    document.getElementById(field).hidden = !learning;
    document.getElementById(input).disabled = !learning;
    document.getElementById(input).required = learning;
  }
}
document.getElementById("changeModeButton").addEventListener("click", () => {
  selectedMode = null;
  formError.hidden = true;
  showOnly(modeView);
});
hapticIntensity.addEventListener("input", updateHapticIntensity);

function updateTrialSummary() {
  const left = Number.parseInt(document.getElementById("leftTrials").value, 10);
  const right = Number.parseInt(
    document.getElementById("rightTrials").value,
    10,
  );
  const learning = selectedMode === "self_learning";
  const total = learning ? Math.max(0, Number(document.getElementById("learningTrials").value))
    : Math.max(0, (Number.isFinite(left) ? left : 0)) +
      Math.max(0, (Number.isFinite(right) ? right : 0));
  const breakEvery = Number.parseInt(
    document.getElementById("breakEveryTrials").value,
    10,
  );
  const validBreakEvery =
    Number.isFinite(breakEvery) && breakEvery > 0 ? breakEvery : 20;
  const breakCount = total > 0
    ? Math.floor((total - 1) / validBreakEvery)
    : 0;
  const totalSeconds = total * (learning ? 10 : TRIAL_SECONDS) + breakCount * BREAK_SECONDS;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  document.getElementById("trialSummary").textContent =
    `共 ${total} 个 trial，组间休息 ${breakCount} 次，最短约 ${minutes} 分 ${String(seconds).padStart(2, "0")} 秒`;
}

document.getElementById("leftTrials").addEventListener("input", updateTrialSummary);
document.getElementById("rightTrials").addEventListener("input", updateTrialSummary);
document.getElementById("learningTrials").addEventListener("input", updateTrialSummary);
document
  .getElementById("breakEveryTrials")
  .addEventListener("input", updateTrialSummary);

async function apiRequest(url, options = {}) {
  const response = await fetch(url, options);
  if (response.ok) {
    return response.json();
  }
  let message = `HTTP ${response.status}`;
  try {
    const body = await response.json();
    message = body.detail || message;
  } catch (_error) {
    // 保留 HTTP 状态作为错误信息。
  }
  throw new Error(message);
}

function prepareAudio() {
  const AudioContextClass = window.AudioContext;
  if (!AudioContextClass) {
    return;
  }
  if (!audioContext) {
    audioContext = new AudioContextClass();
  }
  if (audioContext.state === "suspended") {
    audioContext.resume().catch(() => {});
  }
}

function playChime() {
  prepareAudio();
  if (!audioContext || audioContext.state !== "running") {
    return;
  }
  const now = audioContext.currentTime;
  const gain = audioContext.createGain();
  const oscillator = audioContext.createOscillator();
  oscillator.type = "sine";
  oscillator.frequency.setValueAtTime(880, now);
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(0.28, now + 0.015);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.55);
  oscillator.connect(gain);
  gain.connect(audioContext.destination);
  oscillator.start(now);
  oscillator.stop(now + 0.58);
}

sessionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.hidden = true;
  confirmButton.disabled = true;
  confirmButton.textContent = "正在确认";

  const payload = {
    experiment_mode: selectedMode,
    haptic_intensity: Number(hapticIntensity.value),
    subject_id: document.getElementById("subjectId").value,
    experiment_date: document.getElementById("experimentDate").value,
    description: document.getElementById("description").value,
    left_trials: Number.parseInt(
      document.getElementById("leftTrials").value,
      10,
    ),
    right_trials: Number.parseInt(
      document.getElementById("rightTrials").value,
      10,
    ),
    break_every_trials: Number.parseInt(
      document.getElementById("breakEveryTrials").value,
      10,
    ),
    output_dir: document.getElementById("outputDir").value,
    learning_trials: Number(document.getElementById("learningTrials").value),
    teacher_model_path: document.getElementById("teacherModelPath").value,
  };

  try {
    await apiRequest("/api/session/configure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    hapticRequestError = "";
    modelRequestError = "";
  } catch (error) {
    formError.textContent = error.message;
    formError.hidden = false;
  } finally {
    confirmButton.disabled = false;
    confirmButton.textContent = "确认并连接设备";
  }
});

startButton.addEventListener("click", async () => {
  prepareAudio();
  startButton.disabled = true;
  try {
    await apiRequest("/api/session/start", { method: "POST" });
  } catch (error) {
    deviceMessageText.textContent = error.message;
    deviceMessage.hidden = false;
  }
});

breakToggleButton.addEventListener("click", async () => {
  prepareAudio();
  breakToggleButton.disabled = true;
  try {
    await apiRequest("/api/session/break/toggle", { method: "POST" });
  } catch (error) {
    deviceMessageText.textContent = error.message;
    deviceMessage.hidden = false;
  }
});

async function beginNewSession(chooseAgain = true) {
  try {
    await apiRequest("/api/session/new", { method: "POST" });
    if (chooseAgain) {
      selectedMode = null;
    }
    hapticRequestError = "";
    modelRequestError = "";
    clearPreview();
    showOnly(selectedMode ? informationView : modeView);
  } catch (error) {
    deviceMessageText.textContent = error.message;
    deviceMessage.hidden = false;
  }
}

editSessionButton.addEventListener("click", () => beginNewSession(false));
newSessionButton.addEventListener("click", () => beginNewSession());
completeNewButton.addEventListener("click", () => beginNewSession());

hapticTestButton.addEventListener("click", async () => {
  hapticTestButton.disabled = true;
  hapticRequestError = "";
  try {
    await apiRequest("/api/haptics/test", { method: "POST" });
  } catch (error) {
    hapticRequestError = error.message;
    hapticMessage.textContent = hapticRequestError;
  }
});

document.getElementById("modelTestButton").addEventListener("click", async () => {
  modelRequestError = "";
  document.getElementById("modelTestButton").disabled = true;
  try {
    await apiRequest("/api/self-learning/test", { method: "POST" });
  } catch (error) {
    modelRequestError = error.message;
    document.getElementById("modelMessage").textContent = modelRequestError;
  }
});
for (const action of ["start", "stop", "reset"]) {
  document.getElementById(`preview${action[0].toUpperCase() + action.slice(1)}Button`)
    .addEventListener("click", async () => {
      try {
        await apiRequest(`/api/self-learning/preview/${action}`, { method: "POST" });
        modelRequestError = "";
      } catch (error) {
        modelRequestError = error.message;
        document.getElementById("modelMessage").textContent = modelRequestError;
      }
    });
}

document.addEventListener("keydown", async (event) => {
  if (event.key !== "Escape" || !latestState) {
    return;
  }
  if (!ACTIVE_STAGES.has(latestState.stage) && !latestState.haptics?.busy) {
    return;
  }
  event.preventDefault();
  try {
    await apiRequest("/api/session/abort", { method: "POST" });
  } catch (_error) {
    // 状态广播会显示服务端最终结果。
  }
});

function updateState(state) {
  latestState = state;
  feedbackReceivedAt = performance.now();
  deviceStatus.dataset.connected = String(state.device_connected);
  deviceStatusText.textContent = state.device_status;
  packetStatus.textContent = `Malformed packets: ${state.malformed_packets}`;
  if (state.experiment_mode) {
    selectedMode = state.experiment_mode;
    document.getElementById("selectedMode").textContent = selectedMode;
    document.getElementById("hapticIntensityField").hidden = selectedMode !== "tactile_MI";
    hapticIntensity.value = String(state.haptic_intensity);
    updateHapticIntensity();
    updateModeFields();
  }
  hapticPanel.hidden = !state.haptics;
  const feedback = state.feedback;
  document.getElementById("modelStatus").hidden = !feedback;
  document.getElementById("modelPanel").hidden = !feedback;
  document.getElementById("feedbackPreview").hidden = !feedback || state.stage !== "device_check";
  if (feedback) {
    document.getElementById("modelStatus").dataset.connected = String(feedback.ready);
    document.getElementById("modelStatusText").textContent = feedback.status;
    document.getElementById("modelTestButton").disabled = !state.can_test_model;
    document.getElementById("modelTestButton").textContent = feedback.busy ? "模型测试中…" : "模型测试";
    const test = feedback.test_result;
    document.getElementById("modelMessage").textContent = modelRequestError || feedback.error ||
      (test ? `${test.mock ? "模拟输出 · " : ""}10 条平均 ${test.mean_ms.toFixed(2)} ms/条 · 最大 ${test.max_ms.toFixed(2)} ms · ${test.within_interval ? "低于推理间隔" : "超过推理间隔"}` :
        "模型测试：10 条随机原始窗口，包含 40 Hz 低通、标准化和分类，不含加载。");
    document.getElementById("previewStartButton").disabled =
      !feedback.ready || feedback.busy || feedback.enabled || !state.device_connected;
    document.getElementById("previewStopButton").disabled = !feedback.enabled;
    document.getElementById("previewResetButton").disabled = !feedback.ready || feedback.busy;
  }
  if (state.haptics) {
    for (const motor of state.haptics.motors) {
      const side = motor.motor_id === 0 ? "left" : "right";
      document.getElementById(`${side}HapticStatus`).dataset.connected =
        String(motor.connected);
      document.getElementById(`${side}HapticText`).textContent = motor.status;
    }
    hapticTestButton.disabled = !state.can_test_haptics;
    hapticTestButton.textContent = state.haptics.activity === "test"
      ? "震动测试中…" : "震动测试";
    hapticMessage.textContent = hapticRequestError || state.haptics.message ||
      `高档 ${Number(state.haptic_intensity).toFixed(2)} · 双侧测试 2 秒`;
  }
  if (
    lastChimeSerial === null
    || state.break_chime_serial < lastChimeSerial
  ) {
    lastChimeSerial = state.break_chime_serial;
  } else if (state.break_chime_serial > lastChimeSerial) {
    lastChimeSerial = state.break_chime_serial;
    playChime();
  }

  if (state.stage === "information") {
    showOnly(selectedMode ? informationView : modeView);
    return;
  }

  if (state.stage === "device_check") {
    showOnly(deviceView);
    deviceKicker.textContent = "SIGNAL CHECK";
    deviceTitle.textContent = "设备与信号检查";
    breakTimer.hidden = true;
    breakToggleButton.hidden = true;
    startButton.hidden = false;
    editSessionButton.hidden = false;
    newSessionButton.hidden = false;
    document.getElementById("sessionSubject").textContent =
      `${state.experiment_mode} · Subject: ${state.subject_id}`;
    document.getElementById("sessionTrials").textContent =
      `Trials: ${state.total_trials} · Break every ${state.break_every_trials}`;
    document.getElementById("sessionPath").textContent = state.output_dir;
    startButton.disabled = !state.can_start;
    editSessionButton.disabled = false;
    if (state.last_message) {
      deviceMessageText.textContent = state.last_message;
      deviceMessage.hidden = false;
    } else {
      deviceMessage.hidden = true;
    }
    return;
  }

  if (state.stage === "break_monitor") {
    showOnly(deviceView);
    deviceKicker.textContent = "INTER-BLOCK REST";
    deviceTitle.textContent = "闭眼休息与信号检查";
    document.getElementById("sessionSubject").textContent =
      `${state.experiment_mode} · Subject: ${state.subject_id}`;
    document.getElementById("sessionTrials").textContent =
      `Completed: ${state.completed_trials} / ${state.total_trials}`;
    document.getElementById("sessionPath").textContent = state.output_dir;
    breakTimer.hidden = false;
    breakCountdown.textContent = String(state.break_remaining ?? 30);
    breakTimerStatus.textContent = state.break_paused
      ? "倒计时已暂停"
      : (state.haptics?.busy || state.feedback?.busy ? "测试中，计时暂缓" : "闭眼休息");
    startButton.hidden = true;
    editSessionButton.hidden = true;
    breakToggleButton.hidden = false;
    breakToggleButton.textContent = state.break_paused ? "继续" : "暂停";
    breakToggleButton.disabled =
      state.break_paused ? !state.break_can_continue : false;

    if (!state.device_connected) {
      deviceMessageText.textContent =
        "设备已断开，正在自动重连；恢复有效波形后点击“继续”重新计时30秒。";
      deviceMessage.hidden = false;
      newSessionButton.hidden = true;
    } else if (state.break_paused) {
      deviceMessageText.textContent =
        "倒计时已暂停。确认设备与信号正常后点击“继续”，将重新计时30秒。";
      deviceMessage.hidden = false;
      newSessionButton.hidden = true;
    } else {
      deviceMessage.hidden = true;
    }
    return;
  }

  if (ACTIVE_STAGES.has(state.stage)) {
    showOnly(experimentView);
    renderExperiment(state);
    return;
  }

  if (state.stage === "complete") {
    showOnly(completeView);
    renderComplete(state);
  }
}

function renderExperiment(state) {
  preparationContent.hidden = state.stage !== "preparing";
  restContent.hidden = state.stage !== "rest";
  cueContent.hidden = state.stage !== "cue";
  breakInstructionContent.hidden = state.stage !== "break_instruction";
  document.getElementById("selfLearningContent").hidden = state.stage !== "self_learning";
  document.getElementById("selfLearningRestPrompt").hidden = state.experiment_mode !== "self_learning";
  document.getElementById("learningCounter").textContent =
    `第 ${state.trial_id} / ${state.total_trials} 个 trial · 剩余 ${state.remaining_trials} 个`;
  experimentProgress.hidden =
    state.stage === "preparing" || state.stage === "break_instruction";

  if (state.stage === "preparing") {
    preparationCountdown.textContent = String(
      Math.max(0, state.preparation_remaining ?? 0),
    );
  } else if (state.stage === "rest") {
    remainingTrials.textContent =
      `剩余 ${state.remaining_trials} 个 trial`;
  } else if (state.stage === "cue") {
    const isRight = state.cue_direction === "right_hand";
    directionArrow.dataset.direction = isRight ? "right" : "left";
    directionArrow.setAttribute(
      "aria-label",
      isRight ? "Right arrow" : "Left arrow",
    );
  }
  experimentProgressFill.style.transform =
    `scaleX(${Math.max(0, Math.min(1, state.progress))})`;
}

function renderComplete(state) {
  const summary = state.result_summary || {};
  document.getElementById("completePath").textContent =
    `${state.output_dir}/motor_imagery.h5`;
  document.getElementById("recordedSamples").textContent =
    formatInteger(summary.recorded_sample_count);
  document.getElementById("dataCoverage").textContent =
    Number.isFinite(summary.coverage)
      ? `${(summary.coverage * 100).toFixed(3)}%`
      : "--";
  document.getElementById("completedTrials").textContent =
    `${formatInteger(summary.completed_trials)} / ${state.total_trials}`;
  document.getElementById("malformedDuringSession").textContent =
    formatInteger(summary.malformed_packets_during_session);
}

function formatInteger(value) {
  return Number.isFinite(value) ? Number(value).toLocaleString("en-US") : "--";
}

function appendPreview(channels) {
  for (let channel = 0; channel < CHANNEL_COUNT; channel += 1) {
    const target = previewData[channel];
    target.push(...channels[channel]);
    if (target.length > PREVIEW_POINTS) {
      target.splice(0, target.length - PREVIEW_POINTS);
    }
  }
  previewNeedsDraw = true;
  drawPreview();
}

function clearPreview() {
  for (const channel of previewData) {
    channel.length = 0;
  }
  previewNeedsDraw = true;
  drawPreview();
}

function canvasMetrics(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const pixelWidth = Math.round(width * dpr);
  const pixelHeight = Math.round(height * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { context, width, height };
}

function drawPreview() {
  if (!deviceView.hidden && previewNeedsDraw) {
    const { context, width, height } = canvasMetrics(waveformCanvas);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#f8fafb";
    context.fillRect(0, 0, width, height);

    const left = width < 620 ? 48 : 66;
    const right = 14;
    const top = 8;
    const bottom = 28;
    const plotWidth = width - left - right;
    const laneHeight = (height - top - bottom) / CHANNEL_COUNT;

    context.font = "11px system-ui, sans-serif";
    context.textBaseline = "middle";
    for (let channel = 0; channel < CHANNEL_COUNT; channel += 1) {
      const laneTop = top + laneHeight * channel;
      const centerY = laneTop + laneHeight / 2;

      context.fillStyle = channel % 2 ? "#f4f7f8" : "#ffffff";
      context.fillRect(left, laneTop, plotWidth, laneHeight);
      context.strokeStyle = "#dfe5e8";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(left, centerY);
      context.lineTo(width - right, centerY);
      context.stroke();

      context.fillStyle = "#53616a";
      context.fillText(`CH${channel + 1}`, 8, centerY - 8);
      context.fillStyle = "#849098";
      context.fillText("uV", 14, centerY + 9);

      const values = previewData[channel];
      if (values.length < 2) {
        continue;
      }
      context.save();
      context.beginPath();
      context.rect(left, laneTop, plotWidth, laneHeight);
      context.clip();
      context.strokeStyle = COLORS[channel];
      context.lineWidth = 1.15;
      context.beginPath();
      for (let index = 0; index < values.length; index += 1) {
        const x =
          left + (index / Math.max(1, PREVIEW_POINTS - 1)) * plotWidth;
        const clipped = Math.max(
          -DISPLAY_LIMIT,
          Math.min(DISPLAY_LIMIT, values[index]),
        );
        const y =
          centerY - (clipped / DISPLAY_LIMIT) * (laneHeight * 0.42);
        if (index === 0) {
          context.moveTo(x, y);
        } else {
          context.lineTo(x, y);
        }
      }
      context.stroke();
      context.restore();
    }

    context.fillStyle = "#697780";
    context.textAlign = "left";
    context.fillText("-5 s", left, height - 11);
    context.textAlign = "right";
    context.fillText("0 s", width - right, height - 11);
    context.textAlign = "left";
    previewNeedsDraw = false;
  }
}

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${protocol}//${window.location.host}/ws`);

  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.type === "state") {
      updateState(message);
    } else if (message.type === "samples") {
      appendPreview(message.samples);
    }
  });

  socket.addEventListener("close", () => {
    deviceStatus.dataset.connected = "false";
    deviceStatusText.textContent = "网页连接已断开，正在重连";
    startButton.disabled = true;
    hapticTestButton.disabled = true;
    breakToggleButton.disabled = true;
    latestState = null;
    document.getElementById("modelTestButton").disabled = true;
    for (const id of ["previewStartButton", "previewStopButton", "previewResetButton"]) {
      document.getElementById(id).disabled = true;
    }
    window.setTimeout(connectWebSocket, 1000);
  });
}

window.addEventListener("resize", () => {
  previewNeedsDraw = true;
  drawPreview();
});

// 60 Hz 渲染与 25 Hz teacher 推理解耦。服务器位置为权威，
// 帧间仅按最新速度外推，连接或反馈过期后冻结。
function drawFeedbackFrame(now) {
  const feedback = latestState?.feedback;
  if (feedback) {
    const elapsed = Math.max(0, (now - feedbackReceivedAt) / 1000);
    const dt = Math.min(elapsed, 0.1) / feedback.time_scale;
    const x = Math.max(feedback.config.cursor_min_x, Math.min(
      feedback.config.cursor_max_x,
      feedback.cursor_x + (feedback.enabled ? feedback.velocity * dt : 0),
    ));
    for (const id of ["feedbackBall", "previewBall"]) {
      document.getElementById(id).style.left = `${x * 100}%`;
    }
  }
  requestAnimationFrame(drawFeedbackFrame);
}
requestAnimationFrame(drawFeedbackFrame);

updateTrialSummary();
updateHapticIntensity();
connectWebSocket();
