const patientIdInput = document.getElementById("patient-id");
const languageInput = document.getElementById("language");
const preferredProviderSelect = document.getElementById("preferred-provider");
const strictProviderCheckbox = document.getElementById("strict-provider");
const startSessionButton = document.getElementById("start-session");
const micToggleButton = document.getElementById("mic-toggle");
const endSessionButton = document.getElementById("end-session");
const statusLabel = document.getElementById("status");
const fallbackNote = document.getElementById("fallback-note");
const identityStatus = document.getElementById("identity-status");
const identityNote = document.getElementById("identity-note");
const captureState = document.getElementById("capture-state");
const sessionArtifact = document.getElementById("session-artifact");
const conversationProvider = document.getElementById("conversation-provider");
const liveModelName = document.getElementById("live-model-name");
const processingSteps = document.getElementById("processing-steps");
const recordingPill = document.getElementById("recording-pill");
const frameCount = document.getElementById("frame-count");
const speechSeconds = document.getElementById("speech-seconds");
const audioChunks = document.getElementById("audio-chunks");
const micLevelFill = document.getElementById("mic-level-fill");
const micLevelValue = document.getElementById("mic-level-value");
const promptSteps = document.getElementById("prompt-steps");
const transcript = document.getElementById("transcript");
const videoPreview = document.getElementById("camera-preview");
const snapshotCanvas = document.getElementById("snapshot-canvas");
const providers = document.getElementById("providers");
const manualUploadForm = document.getElementById("manual-upload-form");
const manualPatientIdInput = document.getElementById("manual-patient-id");
const manualLanguageInput = document.getElementById("manual-language");
const manualProviderSelect = document.getElementById("manual-provider");
const manualStrictProviderCheckbox = document.getElementById("manual-strict-provider");
const manualVideoInput = document.getElementById("manual-video");
const manualUploadStatus = document.getElementById("manual-upload-status");
const analysisEmpty = document.getElementById("analysis-empty");
const analysisLoading = document.getElementById("analysis-loading");
const analysisLoadingCopy = document.getElementById("analysis-loading-copy");
const analysisResult = document.getElementById("analysis-result");
const analysisSource = document.getElementById("analysis-source");
const assessmentId = document.getElementById("assessment-id");
const resultCreatedAt = document.getElementById("result-created-at");
const riskLabel = document.getElementById("risk-label");
const riskScore = document.getElementById("risk-score");
const riskTier = document.getElementById("risk-tier");
const screeningClassification = document.getElementById("screening-classification");
const finalProvider = document.getElementById("final-provider");
const reviewerConfidence = document.getElementById("reviewer-confidence");
const screeningSummary = document.getElementById("screening-summary");
const fallbackMessage = document.getElementById("fallback-message");
const visitRecommendation = document.getElementById("visit-recommendation");
const futureRiskTrendSummary = document.getElementById("future-risk-trend-summary");
const evidenceForRisk = document.getElementById("evidence-for-risk");
const evidenceAgainstRisk = document.getElementById("evidence-against-risk");
const alternativeExplanations = document.getElementById("alternative-explanations");
const riskControlSuggestions = document.getElementById("risk-control-suggestions");
const visualFindings = document.getElementById("visual-findings");
const bodyFindings = document.getElementById("body-findings");
const voiceFindings = document.getElementById("voice-findings");
const contentFindings = document.getElementById("content-findings");
const qualityFlags = document.getElementById("quality-flags");
const contextNotes = document.getElementById("context-notes");
const providerTrace = document.getElementById("provider-trace");
const disclaimer = document.getElementById("disclaimer");

const RecognitionConstructor =
  window.SpeechRecognition || window.webkitSpeechRecognition || null;

const state = {
  blueprint: null,
  providerSnapshot: null,
  socket: null,
  stream: null,
  audioContext: null,
  playbackCursorTime: 0,
  mediaSource: null,
  processorNode: null,
  muteGain: null,
  isRecording: false,
  liveAutoMode: false,
  openingInjected: false,
  transcriptTurns: [],
  pendingTurns: [],
  assistantDraftElement: null,
  assistantDraftBody: null,
  assistantDraftText: "",
  assistantResponseTimer: null,
  assistantResponseActive: false,
  assistantResponseCancelSent: false,
  micLevel: 0,
  audioChunkCount: 0,
  speechDurationSeconds: 0,
  utteranceCount: 0,
  turnDurations: [],
  frameCaptureTimer: null,
  frameCaptureBusy: false,
  framesCaptured: 0,
  lastFrameSample: null,
  brightnessSamples: [],
  motionSamples: [],
  faceDetector: null,
  faceDetectionBusy: false,
  faceObservationCount: 0,
  faceDetectionHits: 0,
  faceAreaSamples: [],
  recognition: null,
  recognitionActive: false,
  currentRecognitionText: "",
  serverSpeechActive: false,
  serverSpeechStartedAt: 0,
  realtimeAudioPrimed: false,
  identityPreflight: null,
  identityPreflightCheckedAt: null,
  currentPatientId: "",
  currentLanguage: "en",
  sessionStartedAt: null,
  sessionEndedAt: null,
  mediaRecorder: null,
  recordedChunks: [],
  recordedBlob: null,
  recordingMimeType: "",
  recordingStopPromise: null,
  resolveRecordingStop: null,
  rejectRecordingStop: null,
  sessionAutoEndTimer: null,
  wrapUpTimer: null,
  wrapUpRequested: false,
  sessionEndRequested: false,
  uploadInFlight: false,
};

function applyInitialUrlState() {
  const params = new URLSearchParams(window.location.search);
  const patientId = params.get("patient_id");
  const language = params.get("language");

  if (patientId) {
    patientIdInput.value = patientId;
    manualPatientIdInput.value = patientId;
  }
  if (language) {
    languageInput.value = language;
    manualLanguageInput.value = language;
  }
}

function clearNode(node) {
  if (node) {
    node.replaceChildren();
  }
}

function setStatus(text) {
  statusLabel.textContent = text;
}

function setFallback(text) {
  fallbackNote.textContent = text || "";
}

function updateMicLevel(level) {
  const clamped = Math.max(0, Math.min(1, Number(level) || 0));
  state.micLevel = clamped;
  if (micLevelFill) {
    micLevelFill.style.width = `${Math.round(clamped * 100)}%`;
    micLevelFill.parentElement?.setAttribute("aria-valuenow", String(Math.round(clamped * 100)));
  }
  if (micLevelValue) {
    micLevelValue.textContent = `${Math.round(clamped * 100)}%`;
  }
}

function measureSignalLevel(samples) {
  if (!samples?.length) {
    return { rms: 0, peak: 0 };
  }
  let sumSquares = 0;
  let peak = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.abs(samples[index]);
    sumSquares += sample * sample;
    if (sample > peak) {
      peak = sample;
    }
  }
  return {
    rms: Math.sqrt(sumSquares / samples.length),
    peak,
  };
}

async function ensureAudioContextRunning() {
  if (!state.audioContext) {
    return;
  }
  if (state.audioContext.state === "running") {
    return;
  }
  try {
    await state.audioContext.resume();
  } catch (error) {
    applyRecognitionFallbackNote(
      "Microphone processing could not resume, so live speech capture may stay silent until the browser allows audio again.",
    );
    throw error;
  }
}

function currentMaxSessionSeconds() {
  return Number(state.blueprint?.max_session_seconds || 90);
}

function currentMaxReplySeconds() {
  return Number(state.blueprint?.max_reply_seconds || 7);
}

function currentMaxReplyChars() {
  return Number(state.blueprint?.max_reply_chars || 140);
}

function clearAssistantResponseGuard() {
  if (state.assistantResponseTimer) {
    window.clearTimeout(state.assistantResponseTimer);
    state.assistantResponseTimer = null;
  }
  state.assistantResponseCancelSent = false;
}

function clearSessionAutoEndTimer() {
  if (state.sessionAutoEndTimer) {
    window.clearTimeout(state.sessionAutoEndTimer);
    state.sessionAutoEndTimer = null;
  }
}

function clearWrapUpTimer() {
  if (state.wrapUpTimer) {
    window.clearTimeout(state.wrapUpTimer);
    state.wrapUpTimer = null;
  }
}

function requestAssistantResponseCancel(reason) {
  if (
    state.assistantResponseCancelSent ||
    !state.socket ||
    state.socket.readyState !== WebSocket.OPEN
  ) {
    return;
  }
  state.assistantResponseCancelSent = true;
  try {
    state.socket.send(JSON.stringify({ type: "response.cancel" }));
    if (reason) {
      setStatus(reason);
    }
  } catch {
    // Ignore best-effort cancel failures.
  }
}

function scheduleAssistantResponseGuard() {
  clearAssistantResponseGuard();
  const maxReplySeconds = currentMaxReplySeconds();
  if (!Number.isFinite(maxReplySeconds) || maxReplySeconds <= 0) {
    return;
  }
  state.assistantResponseTimer = window.setTimeout(() => {
    requestAssistantResponseCancel("Reply shortened to keep the conversation moving...");
  }, maxReplySeconds * 1000);
}

function scheduleSessionAutoEnd() {
  clearSessionAutoEndTimer();
  const maxSessionSeconds = currentMaxSessionSeconds();
  if (!Number.isFinite(maxSessionSeconds) || maxSessionSeconds <= 0) {
    return;
  }
  state.sessionAutoEndTimer = window.setTimeout(() => {
    if (state.sessionEndRequested || state.wrapUpRequested || state.uploadInFlight) {
      return;
    }
    requestSessionWrapUp();
  }, maxSessionSeconds * 1000);
}

function handleSessionEndFailure(error) {
  analysisLoading.classList.add("hidden");
  analysisEmpty.classList.remove("hidden");
  state.sessionEndRequested = false;
  state.wrapUpRequested = false;
  clearWrapUpTimer();
  setStatus("Analysis failed");
  setFallback(error instanceof Error ? error.message : "Failed to upload the session for analysis.");
}

function requestSessionWrapUp() {
  if (
    state.wrapUpRequested ||
    state.sessionEndRequested ||
    !state.socket ||
    state.socket.readyState !== WebSocket.OPEN
  ) {
    return;
  }

  state.wrapUpRequested = true;
  clearAssistantResponseGuard();
  clearSessionAutoEndTimer();
  setStatus("Conversation limit reached. Wrapping up...");

  const sendWrapUp = () => {
    if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
      return;
    }
    try {
      state.socket.send(JSON.stringify({ type: "reflexion.wrap_up" }));
    } catch {
      // Ignore best-effort wrap-up failures and let the fallback timer end the session.
    }
  };

  if (state.assistantResponseActive) {
    requestAssistantResponseCancel("Conversation limit reached. Wrapping up...");
    window.setTimeout(sendWrapUp, 220);
  } else {
    sendWrapUp();
  }

  clearWrapUpTimer();
  state.wrapUpTimer = window.setTimeout(() => {
    endSessionAndAnalyze().catch(handleSessionEndFailure);
  }, Math.max((currentMaxReplySeconds() + 2) * 1000, 7000));
}

function setIdentityCheck(text, note = "", tone = "secondary") {
  identityStatus.textContent = text || "Not checked";
  identityStatus.classList.remove("status-secondary", "status-danger");
  if (tone === "secondary") {
    identityStatus.classList.add("status-secondary");
  } else if (tone === "danger") {
    identityStatus.classList.add("status-danger");
  }
  identityNote.textContent = note || "We verify the opening face before the realtime interview starts.";
}

function setSessionArtifact(text) {
  sessionArtifact.textContent = text || "No session recording yet.";
}

function setManualUploadStatus(text) {
  manualUploadStatus.textContent = text || "";
}

function setAnalysisLoadingCopy(text) {
  analysisLoadingCopy.textContent = text || "Uploading media and running multimodal assessment...";
}

function formatNumber(value, digits = 2) {
  return value === null || value === undefined || Number.isNaN(value) ? "-" : Number(value).toFixed(digits);
}

function formatPercent(value) {
  return value === null || value === undefined || Number.isNaN(value)
    ? "-"
    : `${Math.round(Number(value) * 100)}%`;
}

function formatDateTime(value) {
  if (!value) {
    return "-";
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString();
}

function formatLiveModelLabel(modelName, voiceName) {
  const cleanModelName = String(modelName || "").trim() || "Unknown";
  const cleanVoiceName = String(voiceName || "").trim();
  return cleanVoiceName ? `${cleanModelName} · ${cleanVoiceName}` : cleanModelName;
}

function formatBytes(bytes) {
  if (!bytes || bytes < 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  const digits = index === 0 ? 0 : 1;
  return `${size.toFixed(digits)} ${units[index]}`;
}

function toTitleCase(value) {
  return String(value || "")
    .split("_")
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

function scrollTranscriptToBottom() {
  transcript.scrollTop = transcript.scrollHeight;
}

function patientTurnCount() {
  return state.transcriptTurns.filter((turn) => turn.role === "patient").length;
}

function currentStage() {
  if (!state.blueprint?.prompt_steps?.length) {
    return null;
  }
  const index = Math.min(patientTurnCount(), state.blueprint.prompt_steps.length - 1);
  return state.blueprint.prompt_steps[index];
}

function renderPromptSteps() {
  clearNode(promptSteps);

  if (!promptSteps || !state.blueprint?.prompt_steps?.length) {
    return;
  }

  const completedCount = patientTurnCount();
  state.blueprint.prompt_steps.forEach((step, index) => {
    const item = document.createElement("li");
    if (index < completedCount) {
      item.classList.add("is-complete");
    } else if (index === completedCount) {
      item.classList.add("is-active");
    }

    const title = document.createElement("div");
    title.className = "step-title";

    const left = document.createElement("strong");
    left.textContent = step.title;

    const right = document.createElement("span");
    right.className = "step-index";
    right.textContent = `0${index + 1}`;

    title.appendChild(left);
    title.appendChild(right);

    const prompt = document.createElement("p");
    prompt.className = "step-prompt";
    prompt.textContent = step.prompt;

    const goal = document.createElement("p");
    goal.className = "step-goal";
    goal.textContent = `Goal: ${step.goal || "Collect the target signal for this stage."}`;

    const rationale = document.createElement("p");
    rationale.className = "step-rationale";
    rationale.textContent = step.rationale;

    const exitTitle = document.createElement("p");
    exitTitle.className = "step-exit-label";
    exitTitle.textContent = "Exit when";

    const exitList = document.createElement("ul");
    exitList.className = "step-exit";
    (step.exit_when || []).forEach((rule) => {
      const exitItem = document.createElement("li");
      exitItem.textContent = rule;
      exitList.appendChild(exitItem);
    });
    if (exitList.children.length === 0) {
      const exitItem = document.createElement("li");
      exitItem.textContent = "Move on after one sufficient answer.";
      exitList.appendChild(exitItem);
    }

    item.appendChild(title);
    item.appendChild(prompt);
    item.appendChild(goal);
    item.appendChild(rationale);
    item.appendChild(exitTitle);
    item.appendChild(exitList);
    promptSteps.appendChild(item);
  });
}

function renderProcessingSteps(items) {
  clearNode(processingSteps);

  const steps = items && items.length ? items : ["Processing steps unavailable."];
  steps.forEach((entry) => {
    const item = document.createElement("li");
    item.textContent = String(entry);
    processingSteps.appendChild(item);
  });
}

function createMessageElement(role, text) {
  const container = document.createElement("article");
  container.className = `message ${role}`;

  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = role === "assistant" ? "Guide" : "Patient";

  const body = document.createElement("p");
  body.textContent = text;

  container.appendChild(label);
  container.appendChild(body);
  transcript.appendChild(container);
  scrollTranscriptToBottom();

  return { container, body };
}

function addTranscriptTurn(role, text, stage = null) {
  const cleanText = String(text || "").trim();
  if (!cleanText) {
    return;
  }

  state.transcriptTurns.push({ role, text: cleanText, stage });
  createMessageElement(role, cleanText);
  renderPromptSteps();
  refreshButtons();
}

function beginAssistantDraft() {
  if (state.assistantDraftElement) {
    return;
  }
  const elements = createMessageElement("assistant", "");
  state.assistantDraftElement = elements.container;
  state.assistantDraftBody = elements.body;
  state.assistantDraftText = "";
}

function updateAssistantDraft(delta) {
  beginAssistantDraft();
  state.assistantDraftText += delta;
  state.assistantDraftBody.textContent = state.assistantDraftText;
  scrollTranscriptToBottom();
}

function finalizeAssistantDraft(text) {
  const finalText = String(text || state.assistantDraftText || "").trim();
  if (!state.assistantDraftElement || !finalText) {
    return;
  }
  state.assistantDraftBody.textContent = finalText;
  state.transcriptTurns.push({ role: "assistant", text: finalText, stage: null });
  if (!state.liveAutoMode) {
    speakText(finalText);
  }
  state.assistantDraftElement = null;
  state.assistantDraftBody = null;
  state.assistantDraftText = "";
  refreshButtons();
}

function renderSimpleList(node, items, emptyText = "None") {
  clearNode(node);

  const values = items && items.length ? items : [emptyText];
  values.forEach((entry) => {
    const item = document.createElement("li");
    item.textContent = String(entry);
    node.appendChild(item);
  });
}

function renderFindingList(node, items) {
  clearNode(node);

  if (!items || items.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No findings returned.";
    node.appendChild(item);
    return;
  }

  items.forEach((entry) => {
    const item = document.createElement("li");
    const label = document.createElement("strong");
    label.textContent = entry.label || "Finding";
    item.appendChild(label);
    item.appendChild(document.createTextNode(`: ${entry.summary || "No summary."}`));

    const details = [];
    if (entry.evidence) {
      details.push(`Evidence: ${entry.evidence}`);
    }
    if (entry.confidence !== null && entry.confidence !== undefined) {
      details.push(`Confidence: ${formatPercent(entry.confidence)}`);
    }
    if (details.length > 0) {
      item.appendChild(document.createTextNode(` (${details.join(" | ")})`));
    }
    node.appendChild(item);
  });
}

function renderProviderTrace(items) {
  clearNode(providerTrace);

  if (!items || items.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No provider trace returned.";
    providerTrace.appendChild(item);
    return;
  }

  items.forEach((entry) => {
    const item = document.createElement("li");
    const status = entry.failure_reason ? `${entry.status} (${entry.failure_reason})` : entry.status;
    item.textContent = `${entry.provider} · attempt ${entry.attempt_order} · ${status} · ${entry.latency_ms} ms`;
    providerTrace.appendChild(item);
  });
}

function renderFormalAssessment(result, sourceLabel) {
  analysisEmpty.classList.add("hidden");
  analysisLoading.classList.add("hidden");
  analysisResult.classList.remove("hidden");

  analysisSource.textContent = sourceLabel;
  assessmentId.textContent = result.assessment_id || "-";
  resultCreatedAt.textContent = formatDateTime(result.created_at);
  riskLabel.textContent = result.risk_label || "-";
  riskScore.textContent = formatNumber(result.risk_score);
  riskTier.textContent = result.risk_tier ? toTitleCase(result.risk_tier) : "-";
  screeningClassification.textContent = result.screening_classification
    ? toTitleCase(result.screening_classification)
    : "-";

  const providerBits = [];
  if (result.provider_meta?.final_provider) {
    providerBits.push(result.provider_meta.final_provider);
  }
  if (result.provider_meta?.model_name) {
    providerBits.push(result.provider_meta.model_name);
  }
  finalProvider.textContent = providerBits.length ? providerBits.join(" · ") : "-";
  reviewerConfidence.textContent = formatPercent(result.reviewer_confidence);

  screeningSummary.textContent = result.screening_summary || "-";
  const qualityWarning = buildAssessmentQualityMessage(result);
  const supplementalMessage = [result.fallback_message, qualityWarning].filter(Boolean).join(" ");
  fallbackMessage.textContent = supplementalMessage;
  fallbackMessage.classList.toggle("hidden", !supplementalMessage);
  visitRecommendation.textContent = result.visit_recommendation || "-";
  futureRiskTrendSummary.textContent = result.future_risk_trend_summary || "-";
  disclaimer.textContent = result.disclaimer || "";

  renderSimpleList(evidenceForRisk, result.evidence_for_risk, "No supporting risk evidence returned.");
  renderSimpleList(
    evidenceAgainstRisk,
    result.evidence_against_risk,
    "No counter-evidence returned.",
  );
  renderSimpleList(
    alternativeExplanations,
    result.alternative_explanations,
    "No alternative explanations returned.",
  );
  renderSimpleList(
    riskControlSuggestions,
    result.risk_control_suggestions,
    "No risk controls returned.",
  );
  renderFindingList(visualFindings, result.visual_findings);
  renderFindingList(bodyFindings, result.body_findings);
  renderFindingList(voiceFindings, result.voice_findings);
  renderFindingList(contentFindings, result.content_findings);
  renderSimpleList(qualityFlags, result.quality_flags, "No quality flags.");
  renderSimpleList(contextNotes, result.context_notes, "No context notes returned.");
  renderProviderTrace(result.provider_trace);
}

function buildProviderOptionsMarkup(providerStatuses, defaultProvider, currentValue) {
  const options = [
    {
      value: "",
      label: `Auto Router (${defaultProvider})`,
      disabled: false,
    },
  ];

  providerStatuses.forEach((provider) => {
    options.push({
      value: provider.provider,
      label: provider.available
        ? provider.mock_mode
          ? `${provider.provider} (mock)`
          : provider.provider
        : `${provider.provider} (unavailable)`,
      disabled: !provider.available,
    });
  });

  return options.map((option) => {
    const element = document.createElement("option");
    element.value = option.value;
    element.textContent = option.label;
    element.disabled = option.disabled;
    if (option.value === currentValue) {
      element.selected = true;
    }
    return element;
  });
}

function preferredBatchProvider(selectValue) {
  const explicitValue = String(selectValue || "").trim();
  if (explicitValue) {
    return explicitValue;
  }
  return "qwen_omni";
}

function syncSelectOptions(selectNode, providerStatuses, defaultProvider, fallbackValue = "") {
  const preservedValue = selectNode.value || fallbackValue;
  clearNode(selectNode);
  buildProviderOptionsMarkup(providerStatuses, defaultProvider, preservedValue).forEach((option) => {
    selectNode.appendChild(option);
  });
  if (![...selectNode.options].some((option) => option.selected)) {
    selectNode.value = "";
  }
}

function renderProviders(providerStatuses, defaultProvider) {
  clearNode(providers);

  providerStatuses.forEach((provider) => {
    const pill = document.createElement("article");
    pill.className = "provider-pill";

    const title = document.createElement("span");
    title.className = "mini-label";
    title.textContent = `Rank ${provider.fallback_rank}`;

    const name = document.createElement("strong");
    name.textContent = provider.provider;

    const stateText = document.createElement("span");
    stateText.className = provider.available ? "is-available" : "is-unavailable";
    stateText.textContent = provider.available
      ? provider.mock_mode
        ? "Mock Ready"
        : "Available"
      : "Unavailable";

    const meta = document.createElement("p");
    meta.className = "subtle";
    meta.textContent = provider.description;

    pill.appendChild(title);
    pill.appendChild(name);
    pill.appendChild(stateText);
    pill.appendChild(meta);
    providers.appendChild(pill);
  });

  syncSelectOptions(preferredProviderSelect, providerStatuses, defaultProvider, "qwen_omni");
  syncSelectOptions(manualProviderSelect, providerStatuses, defaultProvider, "qwen_omni");
}

function resetSessionState() {
  clearAssistantResponseGuard();
  clearSessionAutoEndTimer();
  clearWrapUpTimer();
  updateMicLevel(0);
  clearNode(transcript);
  state.transcriptTurns = [];
  state.pendingTurns = [];
  state.liveAutoMode = false;
  state.isRecording = false;
  state.assistantDraftElement = null;
  state.assistantDraftBody = null;
  state.assistantDraftText = "";
  state.assistantResponseActive = false;
  state.assistantResponseCancelSent = false;
  state.micLevel = 0;
  state.audioChunkCount = 0;
  state.speechDurationSeconds = 0;
  state.utteranceCount = 0;
  state.turnDurations = [];
  state.playbackCursorTime = 0;
  state.framesCaptured = 0;
  state.lastFrameSample = null;
  state.brightnessSamples = [];
  state.motionSamples = [];
  state.faceObservationCount = 0;
  state.faceDetectionHits = 0;
  state.faceAreaSamples = [];
  state.serverSpeechActive = false;
  state.serverSpeechStartedAt = 0;
  state.realtimeAudioPrimed = false;
  state.identityPreflight = null;
  state.identityPreflightCheckedAt = null;
  state.currentRecognitionText = "";
  state.openingInjected = false;
  state.recordedChunks = [];
  state.recordedBlob = null;
  state.recordingMimeType = "";
  state.sessionStartedAt = null;
  state.sessionEndedAt = null;
  state.wrapUpRequested = false;
  state.sessionEndRequested = false;
  updateCaptureMetrics();
  renderPromptSteps();
  analysisEmpty.classList.remove("hidden");
  analysisLoading.classList.add("hidden");
  analysisResult.classList.add("hidden");
  setRecordingVisualState(false);
  setSessionArtifact("No session recording yet.");
  setIdentityCheck("Not checked");
  refreshButtons();
}

function updateCaptureMetrics() {
  frameCount.textContent = String(state.framesCaptured);
  speechSeconds.textContent = state.speechDurationSeconds.toFixed(1);
  audioChunks.textContent = String(state.audioChunkCount);
}

function refreshButtons() {
  const sessionLive = Boolean(state.socket && state.socket.readyState === WebSocket.OPEN);
  startSessionButton.disabled = state.uploadInFlight;
  micToggleButton.disabled = !sessionLive || state.uploadInFlight;
  endSessionButton.disabled = !sessionLive || state.uploadInFlight;

  if (state.liveAutoMode) {
    micToggleButton.textContent = state.isRecording ? "Pause Mic" : "Resume Mic";
  } else {
    micToggleButton.textContent = state.isRecording ? "Stop & Send" : "Start Answer";
  }
}

function setRecordingVisualState(recording) {
  if (!recording) {
    updateMicLevel(0);
  }
  if (state.liveAutoMode) {
    recordingPill.textContent = recording ? "Mic Live" : "Mic Muted";
  } else {
    recordingPill.textContent = recording ? "Recording Answer" : "Standby";
  }
  recordingPill.classList.toggle("is-recording", recording);
  captureState.textContent = recording
    ? state.liveAutoMode
      ? "Streaming audio + frames"
      : "Capturing one guided answer"
    : state.stream
      ? state.liveAutoMode
        ? "Microphone paused"
        : "Ready for next answer"
      : "Camera Offline";
}

function parseJsonResponse(response) {
  return response
    .json()
    .catch(() => ({
      error: "invalid_response",
      message: "The server returned a non-JSON payload.",
    }));
}

function speakText(text) {
  if (!("speechSynthesis" in window) || !text) {
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1;
  utterance.pitch = 1;
  window.speechSynthesis.speak(utterance);
}

function wsBaseUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}`;
}

function mapRecognitionLanguage(rawLanguage) {
  const clean = String(rawLanguage || "").trim().toLowerCase();
  if (!clean || clean === "en") {
    return "en-US";
  }
  if (clean === "zh" || clean === "zh-cn") {
    return "zh-CN";
  }
  if (clean === "yue" || clean === "zh-hk") {
    return "zh-HK";
  }
  return rawLanguage;
}

function applyDetectedLanguage(languageInputValue) {
  const nextLanguage = String(languageInputValue || "").trim();
  if (!nextLanguage || nextLanguage === state.currentLanguage) {
    return;
  }
  state.currentLanguage = nextLanguage;
  languageInput.value = nextLanguage;
  manualLanguageInput.value = nextLanguage;
  if (state.recognitionActive) {
    stopRecognition();
  }
}

function buildTurnFallbackText(stage, recognitionText) {
  const trimmed = recognitionText.trim();
  if (trimmed) {
    return trimmed;
  }

  const stageTitle =
    state.blueprint?.prompt_steps?.find((step) => step.key === stage)?.title || "current prompt";
  return `Patient audio captured for ${stageTitle}.`;
}

function average(values) {
  if (!values || values.length === 0) {
    return null;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function applyRecognitionFallbackNote(message) {
  const base = String(state.blueprint?.fallback_note || "").trim();
  const detail = String(message || "").trim();
  if (!detail) {
    setFallback(base);
    return;
  }
  setFallback(base ? `${base} ${detail}` : detail);
}

function currentVisualMetrics() {
  return {
    faceDetectionRate:
      state.faceObservationCount > 0 ? state.faceDetectionHits / state.faceObservationCount : null,
    averageFaceArea: average(state.faceAreaSamples),
    motionIntensity: average(state.motionSamples),
    meanBrightness: average(state.brightnessSamples),
  };
}

function currentAudioMetrics() {
  return {
    averageTurnSeconds: average(state.turnDurations),
  };
}

function estimateQualityFlags() {
  const flags = [];
  const visual = currentVisualMetrics();
  if (state.speechDurationSeconds < 15) {
    flags.push("limited_speaking_time");
  }
  if (state.framesCaptured < 4) {
    flags.push("limited_visual_sampling");
  }
  if (visual.faceDetectionRate === null) {
    flags.push("face_detection_unavailable");
  } else if (visual.faceDetectionRate < 0.35) {
    flags.push("face_visibility_low");
  }
  if (patientTurnCount() < 3) {
    flags.push("short_structured_interview");
  }
  return flags;
}

function renderIdentityPreflight(result) {
  const status = String(result?.status || "").trim();
  const noteParts = [result?.summary, result?.recommended_action].filter(Boolean);
  const note = noteParts.join(" ");

  if (status === "verified") {
    setIdentityCheck("Verified", note, "secondary");
    return;
  }
  if (status === "mismatch") {
    setIdentityCheck("Mismatch", note, "danger");
    return;
  }
  if (status === "unenrolled") {
    setIdentityCheck(
      result?.requires_reenrollment ? "Face Re-enroll Needed" : "Not Enrolled Yet",
      note,
      "secondary",
    );
    return;
  }
  if (status === "needs-retry") {
    setIdentityCheck("Retry Needed", note);
    return;
  }
  setIdentityCheck("Not checked");
}

async function runIdentityPreflight(patientId) {
  setIdentityCheck(
    "Checking...",
    "Comparing the opening face sample with the enrolled patient face profile.",
    "secondary",
  );

  const imageBase64Samples = await captureIdentitySamples();
  if (imageBase64Samples.length === 0) {
    const result = {
      patient_id: patientId,
      status: "needs-retry",
      can_start_session: false,
      recommended_action: "Camera frames were unavailable. Keep the patient in frame and try again.",
      summary: "We could not capture opening camera frames for identity verification.",
    };
    state.identityPreflight = result;
    state.identityPreflightCheckedAt = new Date().toISOString();
    renderIdentityPreflight(result);
    return result;
  }

  const response = await fetch("/api/clinic/realtime/identity/check", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      patient_id: patientId,
      image_base64_samples: imageBase64Samples,
    }),
  });
  const payload = await parseJsonResponse(response);

  if (!response.ok) {
    throw new Error(payload.message || "Opening identity check failed.");
  }

  state.identityPreflight = payload;
  state.identityPreflightCheckedAt = new Date().toISOString();
  renderIdentityPreflight(payload);
  return payload;
}

async function loadRealtimeStatus() {
  const response = await fetch("/api/clinic/realtime/status");
  const payload = await parseJsonResponse(response);

  if (!response.ok) {
    throw new Error(payload.message || "Failed to load realtime status.");
  }

  state.blueprint = payload;
  conversationProvider.textContent = payload.conversation_provider;
  liveModelName.textContent = formatLiveModelLabel(payload.model_name, payload.selected_voice);
  setFallback(payload.fallback_note || "");
  renderPromptSteps();
  renderProcessingSteps(payload.processing_steps);
}

async function loadProviderStatus() {
  const response = await fetch("/api/providers");
  const payload = await parseJsonResponse(response);

  if (!response.ok) {
    throw new Error(payload.message || "Failed to load provider status.");
  }

  state.providerSnapshot = payload;
  renderProviders(payload.providers, payload.default_provider);
}

async function ensureMediaReady() {
  if (state.stream) {
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser cannot access camera and microphone streams.");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: true,
    video: {
      facingMode: "user",
      width: { ideal: 640 },
      height: { ideal: 480 },
    },
  });

  state.stream = stream;
  videoPreview.srcObject = stream;
  captureState.textContent = "Camera Ready";

  if (!state.audioContext) {
    await initializeAudioPipeline(stream);
  }

  if (!state.faceDetector && "FaceDetector" in window) {
    try {
      state.faceDetector = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
    } catch {
      state.faceDetector = null;
    }
  }
}

async function initializeAudioPipeline(stream) {
  const AudioContextConstructor = window.AudioContext || window.webkitAudioContext;
  state.audioContext = new AudioContextConstructor();
  await ensureAudioContextRunning();

  state.mediaSource = state.audioContext.createMediaStreamSource(stream);
  state.processorNode = state.audioContext.createScriptProcessor(4096, 1, 1);
  state.muteGain = state.audioContext.createGain();
  state.muteGain.gain.value = 0;

  state.processorNode.onaudioprocess = (event) => {
    if (!state.isRecording || !state.socket || state.socket.readyState !== WebSocket.OPEN) {
      updateMicLevel(0);
      return;
    }

    const inputData = event.inputBuffer.getChannelData(0);
    const { rms } = measureSignalLevel(inputData);
    updateMicLevel(Math.max(rms, state.micLevel * 0.78));
    const pcm16 = convertTo16kPcm(inputData, state.audioContext.sampleRate);
    if (!pcm16 || pcm16.length === 0) {
      return;
    }

    state.audioChunkCount += 1;
    state.realtimeAudioPrimed = true;
    updateCaptureMetrics();
    state.socket.send(
      JSON.stringify({
        type: "input_audio_buffer.append",
        audio: bytesToBase64(new Uint8Array(pcm16.buffer)),
      }),
    );
  };

  state.mediaSource.connect(state.processorNode);
  state.processorNode.connect(state.muteGain);
  state.muteGain.connect(state.audioContext.destination);
}

function convertTo16kPcm(channelData, inputSampleRate) {
  if (!channelData?.length) {
    return null;
  }

  const targetSampleRate = 16000;
  const ratio = inputSampleRate / targetSampleRate;
  const length = Math.max(1, Math.round(channelData.length / ratio));
  const pcm = new Int16Array(length);

  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < pcm.length) {
    const nextOffsetBuffer = Math.min(channelData.length, Math.round((offsetResult + 1) * ratio));
    let accumulator = 0;
    let count = 0;

    for (let index = offsetBuffer; index < nextOffsetBuffer; index += 1) {
      accumulator += channelData[index];
      count += 1;
    }

    const sample = count > 0 ? accumulator / count : channelData[offsetBuffer] || 0;
    const clamped = Math.max(-1, Math.min(1, sample));
    pcm[offsetResult] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;

    offsetResult += 1;
    offsetBuffer = nextOffsetBuffer;
  }

  return pcm;
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }
  return window.btoa(binary);
}

function base64ToBytes(base64) {
  const binary = window.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function playAssistantAudio(base64Audio) {
  if (!state.audioContext || !base64Audio) {
    return;
  }

  const bytes = base64ToBytes(base64Audio);
  const pcm16 = new Int16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  if (pcm16.length === 0) {
    return;
  }

  const sampleRate = 24000;
  const audioBuffer = state.audioContext.createBuffer(1, pcm16.length, sampleRate);
  const channel = audioBuffer.getChannelData(0);
  for (let index = 0; index < pcm16.length; index += 1) {
    channel[index] = pcm16[index] / 0x8000;
  }

  const source = state.audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(state.audioContext.destination);
  const startAt = Math.max(state.audioContext.currentTime, state.playbackCursorTime || 0);
  source.start(startAt);
  state.playbackCursorTime = startAt + audioBuffer.duration;
}

function startFrameCaptureLoop() {
  stopFrameCaptureLoop();
  state.frameCaptureTimer = window.setInterval(() => {
    captureFrame().catch(() => {});
  }, 1000);
}

function stopFrameCaptureLoop() {
  if (state.frameCaptureTimer) {
    window.clearInterval(state.frameCaptureTimer);
    state.frameCaptureTimer = null;
  }
}

function wait(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

async function waitForVideoReady(timeoutMs = 2500) {
  if (videoPreview.videoWidth && videoPreview.videoHeight) {
    return;
  }

  await new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("Camera preview did not become ready in time."));
    }, timeoutMs);

    function cleanup() {
      window.clearTimeout(timeout);
      videoPreview.removeEventListener("loadeddata", onReady);
      videoPreview.removeEventListener("canplay", onReady);
      videoPreview.removeEventListener("playing", onReady);
    }

    function onReady() {
      if (!videoPreview.videoWidth || !videoPreview.videoHeight) {
        return;
      }
      cleanup();
      resolve();
    }

    videoPreview.addEventListener("loadeddata", onReady);
    videoPreview.addEventListener("canplay", onReady);
    videoPreview.addEventListener("playing", onReady);
  });
}

function captureSnapshotBase64() {
  if (!videoPreview.videoWidth || !videoPreview.videoHeight) {
    return null;
  }

  const context = snapshotCanvas.getContext("2d", { willReadFrequently: true });
  snapshotCanvas.width = 480;
  snapshotCanvas.height = 360;
  context.drawImage(videoPreview, 0, 0, snapshotCanvas.width, snapshotCanvas.height);

  return {
    imageData: context.getImageData(0, 0, snapshotCanvas.width, snapshotCanvas.height),
    base64: snapshotCanvas.toDataURL("image/jpeg", 0.72).split(",")[1],
  };
}

async function captureIdentitySamples(sampleCount = 4, delayMs = 140) {
  await waitForVideoReady();

  const samples = [];
  for (let index = 0; index < sampleCount; index += 1) {
    const snapshot = captureSnapshotBase64();
    if (snapshot?.base64) {
      samples.push(snapshot.base64);
    }
    if (index < sampleCount - 1) {
      await wait(delayMs);
    }
  }
  return samples;
}

async function captureFrame() {
  if (
    state.frameCaptureBusy ||
    !state.isRecording ||
    !state.socket ||
    state.socket.readyState !== WebSocket.OPEN ||
    !videoPreview.videoWidth ||
    !videoPreview.videoHeight
  ) {
    return;
  }

  state.frameCaptureBusy = true;
  try {
    const snapshot = captureSnapshotBase64();
    if (!snapshot) {
      return;
    }

    updateVisualMetrics(snapshot.imageData);
    state.framesCaptured += 1;
    updateCaptureMetrics();
    if (state.liveAutoMode && state.realtimeAudioPrimed) {
      state.socket.send(JSON.stringify({ type: "input_image_buffer.append", image: snapshot.base64 }));
    }

    if (state.faceDetector && !state.faceDetectionBusy) {
      state.faceDetectionBusy = true;
      try {
        const faces = await state.faceDetector.detect(snapshotCanvas);
        state.faceObservationCount += 1;
        if (faces.length > 0) {
          const largest = faces.reduce((best, current) => {
            const currentArea = current.boundingBox.width * current.boundingBox.height;
            const bestArea = best.boundingBox.width * best.boundingBox.height;
            return currentArea > bestArea ? current : best;
          }, faces[0]);
          state.faceDetectionHits += 1;
          state.faceAreaSamples.push(
            (largest.boundingBox.width * largest.boundingBox.height) /
              (snapshotCanvas.width * snapshotCanvas.height),
          );
        }
      } catch {
        state.faceDetector = null;
      } finally {
        state.faceDetectionBusy = false;
      }
    }
  } finally {
    state.frameCaptureBusy = false;
  }
}

function updateVisualMetrics(imageData) {
  const { data } = imageData;
  const sample = [];
  let brightnessTotal = 0;
  let count = 0;

  for (let index = 0; index < data.length; index += 64) {
    const gray = (data[index] + data[index + 1] + data[index + 2]) / 3;
    sample.push(gray);
    brightnessTotal += gray;
    count += 1;
  }

  const brightness = count > 0 ? brightnessTotal / count / 255 : 0.5;
  state.brightnessSamples.push(brightness);

  if (state.lastFrameSample) {
    let delta = 0;
    for (let index = 0; index < sample.length; index += 1) {
      delta += Math.abs(sample[index] - (state.lastFrameSample[index] || 0));
    }
    state.motionSamples.push(Math.min(1, delta / sample.length / 255));
  }

  state.lastFrameSample = sample;
}

function ensureRecognition() {
  if (!RecognitionConstructor || state.recognition) {
    return;
  }

  const recognition = new RecognitionConstructor();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  recognition.onresult = (event) => {
    let transcriptText = "";
    for (let index = 0; index < event.results.length; index += 1) {
      transcriptText += event.results[index][0].transcript;
    }
    state.currentRecognitionText = transcriptText.trim();
  };

  recognition.onend = () => {
    state.recognitionActive = false;
    if (state.isRecording) {
      window.setTimeout(() => {
        startRecognition();
      }, 120);
    }
  };

  recognition.onerror = (event) => {
    state.recognitionActive = false;
    const errorCode = String(event?.error || "").trim();
    if (errorCode === "not-allowed" || errorCode === "service-not-allowed") {
      applyRecognitionFallbackNote(
        "Browser speech recognition is blocked for this page, so guided-mode transcript capture may be incomplete.",
      );
      return;
    }
    if (errorCode) {
      applyRecognitionFallbackNote(
        `Browser speech recognition reported "${errorCode}", so guided-mode transcript capture may be incomplete.`,
      );
    }
  };

  state.recognition = recognition;
}

function startRecognition() {
  ensureRecognition();
  if (!RecognitionConstructor) {
    applyRecognitionFallbackNote(
      "This browser does not expose SpeechRecognition, so guided-mode transcript capture falls back to placeholders unless live relay is enabled.",
    );
    return;
  }
  if (!state.recognition || state.recognitionActive) {
    return;
  }

  state.currentRecognitionText = "";
  state.recognition.lang = mapRecognitionLanguage(state.currentLanguage);

  try {
    state.recognition.start();
    state.recognitionActive = true;
  } catch {
    state.recognitionActive = false;
    applyRecognitionFallbackNote(
      "Browser speech recognition could not start, so guided-mode transcript capture may be incomplete.",
    );
  }
}

function stopRecognition() {
  if (!state.recognition || !state.recognitionActive) {
    return;
  }
  try {
    state.recognition.stop();
  } catch {
    state.recognitionActive = false;
  }
}

function chooseRecordingMimeType() {
  if (typeof MediaRecorder === "undefined") {
    return "";
  }
  const candidates = [
    "video/mp4;codecs=h264,aac",
    "video/mp4;codecs=avc1.42E01E,mp4a.40.2",
    "video/mp4",
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm",
  ];
  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) || "";
}

function extensionForMimeType(mimeType) {
  if (mimeType.includes("mp4")) {
    return "mp4";
  }
  if (mimeType.includes("ogg")) {
    return "ogg";
  }
  return "webm";
}

function buildRecordingFilename(mimeType) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const extension = extensionForMimeType(mimeType || "video/webm");
  return `${state.currentPatientId || "patient-session"}-${timestamp}.${extension}`;
}

function prepareSessionRecorder() {
  if (typeof MediaRecorder === "undefined") {
    throw new Error("This browser does not support MediaRecorder for full-session capture.");
  }
  if (!state.stream) {
    throw new Error("Camera and microphone stream is not ready yet.");
  }

  const mimeType = chooseRecordingMimeType();
  const options = {
    videoBitsPerSecond: 1_200_000,
    audioBitsPerSecond: 96_000,
  };
  if (mimeType) {
    options.mimeType = mimeType;
  }
  const recorder = new MediaRecorder(state.stream, options);

  state.mediaRecorder = recorder;
  state.recordedChunks = [];
  state.recordedBlob = null;
  state.recordingMimeType = recorder.mimeType || mimeType || "video/webm";
  state.recordingStopPromise = null;
  state.resolveRecordingStop = null;
  state.rejectRecordingStop = null;

  recorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      state.recordedChunks.push(event.data);
    }
  };

  recorder.onstop = () => {
    const blob =
      state.recordedChunks.length > 0
        ? new Blob(state.recordedChunks, { type: state.recordingMimeType })
        : null;
    state.recordedBlob = blob;
    if (state.resolveRecordingStop) {
      state.resolveRecordingStop(blob);
    }
    state.recordingStopPromise = null;
    state.resolveRecordingStop = null;
    state.rejectRecordingStop = null;
  };

  recorder.onerror = (event) => {
    const error = event.error || new Error("MediaRecorder failed.");
    state.recordedBlob = null;
    if (state.rejectRecordingStop) {
      state.rejectRecordingStop(error);
    }
    state.recordingStopPromise = null;
    state.resolveRecordingStop = null;
    state.rejectRecordingStop = null;
  };
}

function startSessionRecording() {
  prepareSessionRecorder();
  state.sessionStartedAt = new Date().toISOString();
  state.mediaRecorder.start();
  state.recordingStopPromise = new Promise((resolve, reject) => {
    state.resolveRecordingStop = resolve;
    state.rejectRecordingStop = reject;
  });
  setSessionArtifact("Recording full session locally for final upload...");
}

async function stopSessionRecording() {
  if (!state.mediaRecorder) {
    return state.recordedBlob;
  }
  if (state.mediaRecorder.state === "inactive") {
    return state.recordedBlob;
  }

  const stopPromise = state.recordingStopPromise || Promise.resolve(state.recordedBlob);
  state.mediaRecorder.stop();
  return stopPromise;
}

function openRealtimeSocket(patientId, language) {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(
      `${wsBaseUrl()}/api/clinic/realtime/ws?patient_id=${encodeURIComponent(patientId)}&language=${encodeURIComponent(language)}`,
    );

    socket.onopen = () => {
      state.socket = socket;
      state.realtimeAudioPrimed = false;
      refreshButtons();
      resolve(socket);
    };

    socket.onmessage = (event) => {
      try {
        handleRealtimeEvent(JSON.parse(event.data));
      } catch (error) {
        setStatus("Realtime message error");
        setFallback(
          error instanceof Error ? error.message : "Received an invalid realtime message.",
        );
      }
    };

    socket.onerror = () => {
      reject(new Error("Realtime socket failed to connect."));
    };

    socket.onclose = (event) => {
      if (state.socket === socket) {
        state.socket = null;
      }
      clearAssistantResponseGuard();
      clearSessionAutoEndTimer();
      clearWrapUpTimer();
      const code = Number(event?.code || 0);
      const reason = String(event?.reason || "").trim();
      const detailParts = [];
      if (code) {
        detailParts.push(`code ${code}`);
      }
      if (reason) {
        detailParts.push(reason);
      }
      if (event && !event.wasClean) {
        detailParts.push("unclean close");
      }
      if (detailParts.length > 0) {
        setFallback(`Realtime socket closed: ${detailParts.join(", ")}`);
      }
      if (!state.uploadInFlight) {
        setStatus("Realtime disconnected");
      }
      disableLiveAutoMode();
      refreshButtons();
    };
  });
}

async function closeRealtimeSocket() {
  clearAssistantResponseGuard();
  clearSessionAutoEndTimer();
  clearWrapUpTimer();
  if (!state.socket) {
    return;
  }
  try {
    state.socket.send(JSON.stringify({ type: "reflexion.close" }));
  } catch {
    // Ignore best-effort close notifications.
  }
  state.socket.close();
  state.socket = null;
}

function queueTranscriptFallback(stage, fallbackText, timeoutMs) {
  const pendingTurn = {
    stage,
    fallbackText,
    fulfilled: false,
  };
  state.pendingTurns.push(pendingTurn);

  window.setTimeout(() => {
    if (pendingTurn.fulfilled) {
      return;
    }
    pendingTurn.fulfilled = true;
    addTranscriptTurn("patient", pendingTurn.fallbackText, pendingTurn.stage);
  }, timeoutMs);
}

function enableLiveAutoMode() {
  state.liveAutoMode = true;
  state.isRecording = true;
  ensureAudioContextRunning().catch(() => {});
  startFrameCaptureLoop();
  startRecognition();
  setRecordingVisualState(true);
  setStatus("Mic live, waiting for speech...");
  refreshButtons();
}

function disableLiveAutoMode() {
  state.liveAutoMode = false;
  state.isRecording = false;
  stopFrameCaptureLoop();
  stopRecognition();
  setRecordingVisualState(false);
  refreshButtons();
}

function toggleLiveMicrophone() {
  if (state.isRecording) {
    state.isRecording = false;
    stopFrameCaptureLoop();
    stopRecognition();
    setRecordingVisualState(false);
    setStatus("Microphone paused");
    refreshButtons();
    return;
  }

  state.isRecording = true;
  startFrameCaptureLoop();
  startRecognition();
  setRecordingVisualState(true);
  setStatus("Mic live, waiting for speech...");
  refreshButtons();
}

function startRecordingTurn() {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
    return;
  }

  state.isRecording = true;
  state.recordingStartedAt = performance.now();
  setRecordingVisualState(true);
  setStatus("Listening for patient answer...");
  startFrameCaptureLoop();
  startRecognition();
  refreshButtons();
}

function stopRecordingTurn() {
  if (!state.isRecording || !state.socket || state.socket.readyState !== WebSocket.OPEN) {
    return;
  }

  state.isRecording = false;
  stopFrameCaptureLoop();
  stopRecognition();

  const durationSeconds = (performance.now() - state.recordingStartedAt) / 1000;
  state.turnDurations.push(durationSeconds);
  state.speechDurationSeconds += durationSeconds;
  state.utteranceCount += 1;
  updateCaptureMetrics();
  setRecordingVisualState(false);
  setStatus("Submitting turn...");

  const stage = currentStage()?.key || null;
  const fallbackText = buildTurnFallbackText(stage, state.currentRecognitionText || "");
  const timeoutMs = state.blueprint?.session_mode === "live_qwen" ? 1800 : 80;
  queueTranscriptFallback(stage, fallbackText, timeoutMs);
  state.currentRecognitionText = "";

  if (state.blueprint?.session_mode !== "live_qwen") {
    state.socket.send(
      JSON.stringify({
        type: "reflexion.patient_turn",
        text: fallbackText,
        stage,
      }),
    );
  }

  state.socket.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
  window.setTimeout(() => {
    if (state.socket && state.socket.readyState === WebSocket.OPEN) {
      state.socket.send(JSON.stringify({ type: "response.create" }));
      setStatus("Generating reply...");
    }
  }, 180);
  refreshButtons();
}

function flushRecognitionDraftIntoTranscript() {
  const fallbackText = buildTurnFallbackText(currentStage()?.key || null, state.currentRecognitionText || "");
  if (!fallbackText) {
    return;
  }

  const lastTurn = state.transcriptTurns[state.transcriptTurns.length - 1];
  if (lastTurn?.role === "patient" && lastTurn.text === fallbackText) {
    return;
  }
  addTranscriptTurn("patient", fallbackText, currentStage()?.key || null);
}

async function toggleRecordingTurn() {
  if (state.liveAutoMode) {
    toggleLiveMicrophone();
    return;
  }
  if (state.isRecording) {
    stopRecordingTurn();
  } else {
    startRecordingTurn();
  }
}

function handleRealtimeEvent(event) {
  const type = event.type;

  if (type === "reflexion.session.ready") {
    state.blueprint = event.session;
    conversationProvider.textContent = event.session.conversation_provider;
    liveModelName.textContent = formatLiveModelLabel(
      event.session.model_name,
      event.session.selected_voice,
    );
    renderProcessingSteps(event.session.processing_steps);
    setFallback(event.session.fallback_note || "");
    renderPromptSteps();
    if (!state.openingInjected) {
      addTranscriptTurn("assistant", event.session.greeting);
      speakText(event.session.greeting);
      state.openingInjected = true;
    }
    if (event.session.session_mode === "live_qwen") {
      state.wrapUpRequested = false;
      scheduleSessionAutoEnd();
      setStatus("Connecting live relay...");
    } else {
      clearSessionAutoEndTimer();
      disableLiveAutoMode();
      setStatus("Guided conversation ready");
    }
    refreshButtons();
    return;
  }

  if (type === "reflexion.session.degraded") {
    clearSessionAutoEndTimer();
    clearAssistantResponseGuard();
    clearWrapUpTimer();
    state.wrapUpRequested = false;
    disableLiveAutoMode();
    state.blueprint = event.session;
    conversationProvider.textContent = event.session.conversation_provider;
    liveModelName.textContent = formatLiveModelLabel(
      event.session.model_name,
      event.session.selected_voice,
    );
    renderProcessingSteps(event.session.processing_steps);
    const reason = String(event.reason || "").trim();
    const note = event.session.fallback_note || "";
    setFallback(reason ? `${note} Reason: ${reason}` : note);
    renderPromptSteps();
    setStatus("Live relay degraded to guided conversation");
    return;
  }

  if (type === "reflexion.voice.selected") {
    if (state.blueprint) {
      state.blueprint.selected_voice = event.voice || state.blueprint.selected_voice;
      state.blueprint.selected_language = event.language || state.blueprint.selected_language;
    }
    liveModelName.textContent = formatLiveModelLabel(
      state.blueprint?.model_name || liveModelName.textContent,
      event.voice,
    );
    applyDetectedLanguage(event.language_input);
    return;
  }

  if (type === "input_audio_buffer.speech_started") {
    if (state.liveAutoMode) {
      state.serverSpeechActive = true;
      state.serverSpeechStartedAt = performance.now();
      state.currentRecognitionText = "";
      window.speechSynthesis?.cancel?.();
      setStatus("Patient speaking...");
    }
    return;
  }

  if (type === "input_audio_buffer.speech_stopped") {
    if (state.liveAutoMode && state.serverSpeechActive) {
      const durationSeconds = (performance.now() - state.serverSpeechStartedAt) / 1000;
      state.serverSpeechActive = false;
      state.turnDurations.push(durationSeconds);
      state.speechDurationSeconds += durationSeconds;
      state.utteranceCount += 1;
      updateCaptureMetrics();
      const stage = currentStage()?.key || null;
      const fallbackText = buildTurnFallbackText(stage, state.currentRecognitionText || "");
      queueTranscriptFallback(stage, fallbackText, 1800);
      setStatus("Turn detected, waiting for reply...");
    }
    return;
  }

  if (type === "input_audio_buffer.committed") {
    if (state.liveAutoMode) {
      setStatus("Turn committed, response in progress...");
    }
    return;
  }

  if (type === "conversation.item.input_audio_transcription.completed") {
    let pending = state.pendingTurns.shift();
    while (pending?.fulfilled) {
      pending = state.pendingTurns.shift();
    }
    const stage = pending?.stage || currentStage()?.key || null;
    const text = String(event.transcript || "").trim() || pending?.fallbackText || "";
    if (pending) {
      pending.fulfilled = true;
    }
    addTranscriptTurn("patient", text, stage);
    return;
  }

  if (type === "response.created") {
    state.assistantResponseActive = true;
    beginAssistantDraft();
    scheduleAssistantResponseGuard();
    return;
  }

  if (type === "response.audio.delta") {
    playAssistantAudio(String(event.delta || ""));
    return;
  }

  if (type === "response.audio_transcript.delta") {
    updateAssistantDraft(String(event.delta || ""));
    if (state.assistantDraftText.trim().length > currentMaxReplyChars()) {
      requestAssistantResponseCancel("Reply shortened to keep the conversation moving...");
    }
    return;
  }

  if (type === "response.audio_transcript.done") {
    clearAssistantResponseGuard();
    finalizeAssistantDraft(String(event.transcript || state.assistantDraftText || ""));
    return;
  }

  if (type === "response.text.delta") {
    updateAssistantDraft(String(event.delta || ""));
    if (state.assistantDraftText.trim().length > currentMaxReplyChars()) {
      requestAssistantResponseCancel("Reply shortened to keep the conversation moving...");
    }
    return;
  }

  if (type === "response.text.done") {
    clearAssistantResponseGuard();
    finalizeAssistantDraft(String(event.text || state.assistantDraftText || ""));
    return;
  }

  if (type === "response.done") {
    state.assistantResponseActive = false;
    clearAssistantResponseGuard();
    finalizeAssistantDraft(state.assistantDraftText);
    if (state.wrapUpRequested && !state.sessionEndRequested) {
      endSessionAndAnalyze().catch(handleSessionEndFailure);
      return;
    }
    if (state.liveAutoMode && state.isRecording) {
      setStatus("Mic live, waiting for speech...");
    }
    return;
  }

  if (type === "session.updated" || type === "session.created") {
    if (state.blueprint?.session_mode === "live_qwen" && !state.liveAutoMode) {
      enableLiveAutoMode();
    }
    if (!state.wrapUpRequested) {
      setStatus("Session ready");
    }
    return;
  }

  if (type === "error") {
    state.assistantResponseActive = false;
    clearAssistantResponseGuard();
    const message = event.error?.message || event.error || "Realtime error.";
    setStatus("Error");
    setFallback(String(message));
  }
}

async function startSession() {
  setStatus("Preparing camera and microphone...");
  setFallback("");

  await closeRealtimeSocket();
  resetSessionState();

  state.currentPatientId = patientIdInput.value.trim() || "patient-001";
  state.currentLanguage = languageInput.value.trim() || "en";

  await ensureMediaReady();
  await ensureAudioContextRunning();

  const identityResult = await runIdentityPreflight(state.currentPatientId);
  if (!identityResult.can_start_session) {
    const blockedStatus = identityResult.requires_patient_reentry ? "Identity mismatch" : "Identity retry needed";
    setStatus(blockedStatus);
    setFallback(identityResult.recommended_action || identityResult.summary || "Identity check blocked the session start.");
    if (identityResult.requires_patient_reentry) {
      patientIdInput.focus();
      patientIdInput.select?.();
    }
    return;
  }

  setStatus(
    identityResult.status === "verified"
      ? "Identity verified. Preparing session..."
      : "No enrolled face yet. Preparing session...",
  );
  setFallback(identityResult.summary || "");

  try {
    startSessionRecording();
  } catch (error) {
    setSessionArtifact(
      error instanceof Error
        ? `${error.message} You can still run a live conversation and use manual upload.`
        : "Session recording is unavailable in this browser.",
    );
  }

  try {
    await openRealtimeSocket(state.currentPatientId, state.currentLanguage);
  } catch (error) {
    if (state.mediaRecorder?.state === "recording") {
      await stopSessionRecording().catch(() => {});
    }
    throw error;
  }

  startSessionButton.textContent = "Restart Session";
  setStatus("Waiting for opening prompt...");
}

function buildErrorMessage(payload, fallbackText) {
  if (!payload) {
    return fallbackText;
  }
  const candidate = payload.message || payload.detail || payload.error;
  let message = fallbackText;
  if (typeof candidate === "string") {
    message = candidate;
  } else if (Array.isArray(candidate)) {
    message = candidate.join("; ");
  } else if (candidate && typeof candidate === "object") {
    try {
      message = JSON.stringify(candidate);
    } catch {
      message = fallbackText;
    }
  }

  const providerTrace = Array.isArray(payload.provider_trace) ? payload.provider_trace : [];
  if (providerTrace.length > 0) {
    const summary = providerTrace
      .map((entry) => {
        const reason = entry.failure_reason ? ` (${entry.failure_reason})` : "";
        return `${entry.provider}${reason}`;
      })
      .join(", ");
    message = `${message} Providers: ${summary}.`;
  }

  if (providerTrace.some((entry) => entry.failure_reason === "unsupported_media")) {
    message +=
      " The recorded session may exceed Qwen's inline batch limit. Shorten the session, lower recording bitrate, install ffmpeg for server-side standardization, or configure Gemini/OpenAI fallbacks.";
  }

  const qwenUnusable = providerTrace.find(
    (entry) => entry.provider === "qwen_omni" && entry.failure_reason === "unusable_result",
  );
  const qwenFlags = Array.isArray(qwenUnusable?.debug_details?.quality_flags)
    ? qwenUnusable.debug_details.quality_flags
    : [];
  if (qwenFlags.length > 0) {
    const advice = [];
    if (qwenFlags.includes("video_too_short")) {
      advice.push("keep the session running longer before ending it");
    }
    if (qwenFlags.includes("speech_unintelligible") || qwenFlags.includes("transcript_unavailable")) {
      advice.push("make sure the patient speaks clearly for a few seconds");
    }
    if (qwenFlags.includes("face_not_visible") || qwenFlags.includes("low_light")) {
      advice.push("keep the patient centered in brighter light");
    }
    if (advice.length > 0) {
      message += ` Qwen judged the clip unusable; ${advice.join(", ")}.`;
    }
  }

  return message;
}

function buildAssessmentQualityMessage(result) {
  const flags = Array.isArray(result?.quality_flags) ? result.quality_flags : [];
  const usability = String(result?.session_usability || "").trim().toLowerCase();
  if (!flags.length && usability !== "unusable") {
    return "";
  }

  const flagLabels = flags.map((flag) => {
    switch (String(flag || "").trim()) {
      case "video_too_short":
        return "the video was too short";
      case "transcript_unavailable":
        return "no transcript was available";
      case "speech_unintelligible":
        return "speech could not be understood";
      case "face_not_visible":
        return "the patient face was not clearly visible";
      case "low_light":
        return "lighting was poor";
      case "limited_speaking_time":
        return "there was very little patient speech";
      default:
        return toTitleCase(flag);
    }
  });

  if (usability === "unusable") {
    const issues = flagLabels.length ? ` Possible quality issues: ${flagLabels.join(", ")}.` : "";
    return `Quality note: this clip was marked unusable for reliable screening interpretation.${issues}`;
  }

  if (flagLabels.length) {
    return `Quality note: review this result with caution because ${flagLabels.join(", ")}.`;
  }

  return "";
}

function buildSessionRecord(filename, blob) {
  const visual = currentVisualMetrics();
  const audio = currentAudioMetrics();
  const endedAt = state.sessionEndedAt || new Date().toISOString();
  const completedStages = [...new Set(state.transcriptTurns.map((turn) => turn.stage).filter(Boolean))];
  const qualityFlagList = estimateQualityFlags();
  const audioQualityScore = Math.min(1, Math.max(0.2, state.speechDurationSeconds / 45));
  const videoQualityScore = Math.min(
    1,
    Math.max(0.2, ((state.framesCaptured / 12) + (visual.faceDetectionRate || 0.35)) / 2),
  );

  return {
    sessionId: `live-${Date.now()}`,
    productMode: "clinic",
    intendedUseVersion: "realtime-batch-demo-v1",
    consent: {
      status: "not-required",
      consentVersion: "demo",
    },
    patient: {
      patientId: state.currentPatientId,
      primaryLanguage: state.currentLanguage,
    },
    site: {
      siteId: "browser-demo",
      siteType: "research",
    },
    device: {
      deviceId: "browser-capture",
      softwareVersion: navigator.userAgent,
      captureProfile: "webcam-microphone",
    },
    acquisition: {
      startedAt: state.sessionStartedAt || endedAt,
      endedAt,
      language: state.currentLanguage,
      tasksCompleted: completedStages,
      operatorId: "browser-demo",
    },
    identityAttribution: {
      openingCheck: state.identityPreflight
        ? {
            checkedAt: state.identityPreflightCheckedAt || endedAt,
            ...state.identityPreflight,
          }
        : null,
    },
    rawArtifacts: {
      video: [
        {
          artifactId: "recorded-session-video",
          uri: `browser-recording://${filename}`,
          createdAt: endedAt,
        },
      ],
      transcript: [
        {
          artifactId: "browser-live-transcript",
          uri: "inline://browser-transcript",
          createdAt: endedAt,
        },
      ],
    },
    derivedFeatures: {
      speech: {
        utteranceCount: state.utteranceCount,
        speechSeconds: state.speechDurationSeconds,
        averageTurnSeconds: audio.averageTurnSeconds,
        transcriptTurns: state.transcriptTurns,
      },
      task: {
        completedStages,
        patientTurns: patientTurnCount(),
      },
      facial: {
        framesCaptured: state.framesCaptured,
        faceDetectionRate: visual.faceDetectionRate,
        averageFaceArea: visual.averageFaceArea,
      },
      interactionTiming: {
        turnDurationsSeconds: state.turnDurations,
        motionIntensity: visual.motionIntensity,
        meanBrightness: visual.meanBrightness,
        recordingBytes: blob?.size || 0,
      },
    },
    qualityControl: {
      usability: patientTurnCount() >= 3 ? "usable" : "usable-with-caveats",
      audioQualityScore,
      videoQualityScore,
      speechDurationSeconds: state.speechDurationSeconds,
      flags: qualityFlagList,
    },
    modelContext: {
      featurePipelineVersion: "browser-capture-v1",
      clinicModelVersion: "provider-mesh-upload-v1",
      datasetVersion: "live-browser-session",
    },
  };
}

async function submitBatchAssessment({
  videoBlob,
  filename,
  patientId,
  language,
  preferredProvider,
  strictProvider,
  sourceLabel,
  sessionRecord,
}) {
  const formData = new FormData();
  formData.append("video", videoBlob, filename);
  formData.append("patient_id", patientId);
  formData.append("language", language);
  formData.append("strict_provider", strictProvider ? "true" : "false");
  if (preferredProvider) {
    formData.append("preferred_provider", preferredProvider);
  }
  if (sessionRecord) {
    formData.append("session_record_json", JSON.stringify(sessionRecord));
  }

  state.uploadInFlight = true;
  refreshButtons();
  analysisEmpty.classList.add("hidden");
  analysisResult.classList.add("hidden");
  analysisLoading.classList.remove("hidden");
  setAnalysisLoadingCopy(
    sourceLabel === "Live Session Upload"
      ? "Uploading the recorded live session and running formal multimodal assessment..."
      : "Uploading video clip and running formal multimodal assessment...",
  );
  setFallback("");

  const response = await fetch("/api/clinic/video/analyze", {
    method: "POST",
    body: formData,
  });
  const payload = await parseJsonResponse(response);
  state.uploadInFlight = false;
  refreshButtons();

  if (!response.ok) {
    analysisLoading.classList.add("hidden");
    analysisEmpty.classList.remove("hidden");
    throw new Error(buildErrorMessage(payload, "Formal multimodal analysis failed."));
  }

  renderFormalAssessment(payload, sourceLabel);
  return payload;
}

async function endSessionAndAnalyze() {
  if (state.sessionEndRequested || state.uploadInFlight) {
    return;
  }
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
    throw new Error("Start a session before ending and analyzing it.");
  }

  state.sessionEndRequested = true;
  clearAssistantResponseGuard();
  clearSessionAutoEndTimer();
  clearWrapUpTimer();
  setStatus("Stopping live session and packaging the recording...");
  setFallback("");
  state.sessionEndedAt = new Date().toISOString();

  if (!state.liveAutoMode && state.isRecording) {
    stopFrameCaptureLoop();
    stopRecognition();
    state.isRecording = false;
    if (state.currentRecognitionText.trim()) {
      flushRecognitionDraftIntoTranscript();
    }
    state.currentRecognitionText = "";
    setRecordingVisualState(false);
  } else if (state.liveAutoMode && state.isRecording) {
    state.isRecording = false;
    stopFrameCaptureLoop();
    stopRecognition();
    setRecordingVisualState(false);
  }

  const recordingPromise = stopSessionRecording();
  await closeRealtimeSocket();

  const blob = await recordingPromise;
  if (!blob || blob.size === 0) {
    throw new Error("No full session recording was produced. Try the manual upload flow instead.");
  }

  state.recordedBlob = blob;
  const filename = buildRecordingFilename(state.recordingMimeType || blob.type);
  setSessionArtifact(`Recorded ${formatBytes(blob.size)} session clip. Uploading now...`);

  const result = await submitBatchAssessment({
    videoBlob: blob,
    filename,
    patientId: state.currentPatientId,
    language: state.currentLanguage,
    preferredProvider: preferredBatchProvider(preferredProviderSelect.value),
    strictProvider: strictProviderCheckbox.checked,
    sourceLabel: "Live Session Upload",
    sessionRecord: buildSessionRecord(filename, blob),
  });

  setStatus("Formal assessment ready");
  setSessionArtifact(`Uploaded recorded session (${formatBytes(blob.size)}) for assessment ${result.assessment_id}.`);
}

async function handleManualUpload(event) {
  event.preventDefault();

  const file = manualVideoInput.files?.[0];
  if (!file) {
    throw new Error("Choose a video file before starting manual analysis.");
  }

  const patientId = manualPatientIdInput.value.trim() || "patient-001";
  const language = manualLanguageInput.value.trim() || "en";
  setManualUploadStatus(`Uploading ${file.name} (${formatBytes(file.size)})...`);

  const result = await submitBatchAssessment({
    videoBlob: file,
    filename: file.name || "manual-upload.webm",
    patientId,
    language,
    preferredProvider: preferredBatchProvider(manualProviderSelect.value),
    strictProvider: manualStrictProviderCheckbox.checked,
    sourceLabel: "Manual Upload",
    sessionRecord: null,
  });

  setStatus("Formal assessment ready");
  setManualUploadStatus(`Completed manual upload analysis for assessment ${result.assessment_id}.`);
}

startSessionButton.addEventListener("click", () => {
  startSession().catch((error) => {
    setStatus("Failed");
    setFallback(error instanceof Error ? error.message : "Failed to start session.");
  });
});

micToggleButton.addEventListener("click", () => {
  toggleRecordingTurn();
});

endSessionButton.addEventListener("click", () => {
  endSessionAndAnalyze().catch((error) => {
    analysisLoading.classList.add("hidden");
    analysisEmpty.classList.remove("hidden");
    setStatus("Analysis failed");
    setFallback(error instanceof Error ? error.message : "Failed to upload the session for analysis.");
  });
});

manualUploadForm.addEventListener("submit", (event) => {
  handleManualUpload(event).catch((error) => {
    analysisLoading.classList.add("hidden");
    analysisEmpty.classList.remove("hidden");
    setStatus("Analysis failed");
    setManualUploadStatus(error instanceof Error ? error.message : "Manual upload failed.");
  });
});

window.addEventListener("beforeunload", () => {
  closeRealtimeSocket().catch(() => {});
});

applyInitialUrlState();

Promise.all([loadRealtimeStatus(), loadProviderStatus()])
  .then(() => {
    renderPromptSteps();
    refreshButtons();
    setSessionArtifact("No session recording yet.");
    setIdentityCheck("Not checked");
    setManualUploadStatus(
      "Upload a video clip to run the same batch multimodal analysis without a live session.",
    );
  })
  .catch((error) => {
    setStatus("Failed");
    setFallback(error instanceof Error ? error.message : "Failed to load the demo.");
  });
