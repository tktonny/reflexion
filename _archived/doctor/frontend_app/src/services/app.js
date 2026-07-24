const refreshButton = document.getElementById("refresh-dashboard");
const doctorStatus = document.getElementById("doctor-status");
const totalAssessments = document.getElementById("total-assessments");
const totalPatients = document.getElementById("total-patients");
const highRiskAssessments = document.getElementById("high-risk-assessments");
const watchlistAssessments = document.getElementById("watchlist-assessments");
const averageRiskScore = document.getElementById("average-risk-score");
const usableSessions = document.getElementById("usable-sessions");
const recentAssessmentRows = document.getElementById("recent-assessment-rows");
const patientSummaryRows = document.getElementById("patient-summary-rows");
const providerMix = document.getElementById("provider-mix");

function formatDateTime(value) {
  if (!value) {
    return "-";
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString();
}

function formatPercent(value) {
  return value === null || value === undefined || Number.isNaN(value)
    ? "-"
    : `${Math.round(Number(value) * 100)}%`;
}

function toTitleCase(value) {
  return String(value || "-")
    .split("_")
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

function signalFromRisk(riskTier) {
  if (riskTier === "high") {
    return ["pill", "pill-urgent"];
  }
  if (riskTier === "medium") {
    return ["pill", "pill-watch"];
  }
  return ["pill", "pill-stable"];
}

function setLoadingState(copy) {
  doctorStatus.textContent = copy;
}

function clearNode(node) {
  node.replaceChildren();
}

function appendEmptyRow(node, message, columns) {
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.colSpan = columns;
  cell.className = "empty-state";
  cell.setAttribute("data-label", "Status");
  cell.textContent = message;
  row.appendChild(cell);
  node.appendChild(row);
}

function appendCell(row, label, value) {
  const cell = document.createElement("td");
  cell.setAttribute("data-label", label);
  if (value instanceof Node) {
    cell.appendChild(value);
  } else {
    cell.textContent = value;
  }
  row.appendChild(cell);
}

function renderMetrics(payload) {
  totalAssessments.textContent = payload.total_assessments ?? "-";
  totalPatients.textContent = payload.total_patients ?? "-";
  highRiskAssessments.textContent = payload.high_risk_assessments ?? "-";
  watchlistAssessments.textContent = payload.watchlist_assessments ?? "-";
  averageRiskScore.textContent = formatPercent(payload.average_risk_score);
  usableSessions.textContent = payload.usable_sessions ?? "-";
}

function renderRecentAssessments(items) {
  clearNode(recentAssessmentRows);

  if (!items?.length) {
    appendEmptyRow(recentAssessmentRows, "No assessments have been written yet.", 6);
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("tr");

    const patientCell = document.createElement("strong");
    patientCell.textContent = item.patient_id;

    const createdCell = formatDateTime(item.created_at);

    const pill = document.createElement("span");
    pill.className = signalFromRisk(item.risk_tier).join(" ");
    pill.textContent = toTitleCase(item.risk_tier || "low");

    const classificationCell = toTitleCase(item.screening_classification || "pending");

    const riskCell = formatPercent(item.risk_score);

    const actions = document.createElement("div");
    actions.className = "action-links";

    const careLink = document.createElement("a");
    careLink.href = `/care?patient_id=${encodeURIComponent(item.patient_id)}`;
    careLink.textContent = "Care view";

    const clinicLink = document.createElement("a");
    clinicLink.href = `/clinic?patient_id=${encodeURIComponent(item.patient_id)}`;
    clinicLink.textContent = "Clinic";

    actions.appendChild(careLink);
    actions.appendChild(clinicLink);

    appendCell(row, "Patient", patientCell);
    appendCell(row, "Created", createdCell);
    appendCell(row, "Tier", pill);
    appendCell(row, "Classification", classificationCell);
    appendCell(row, "Risk", riskCell);
    appendCell(row, "Actions", actions);
    recentAssessmentRows.appendChild(row);
  });
}

function renderPatientSummaries(items) {
  clearNode(patientSummaryRows);

  if (!items?.length) {
    appendEmptyRow(patientSummaryRows, "No patient rollup is available yet.", 5);
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("tr");

    const link = document.createElement("a");
    link.href = `/care?patient_id=${encodeURIComponent(item.patient_id)}`;
    link.textContent = item.patient_id;

    const sessionsCell = String(item.session_count ?? "-");

    const latestCell = formatDateTime(item.latest_assessment_at);

    const pill = document.createElement("span");
    pill.className = signalFromRisk(item.latest_risk_tier).join(" ");
    pill.textContent = toTitleCase(item.latest_risk_tier || "low");

    const averageCell = formatPercent(item.average_risk_score);

    appendCell(row, "Patient", link);
    appendCell(row, "Sessions", sessionsCell);
    appendCell(row, "Latest", latestCell);
    appendCell(row, "Tier", pill);
    appendCell(row, "Average Risk", averageCell);
    patientSummaryRows.appendChild(row);
  });
}

function renderProviderMix(items) {
  clearNode(providerMix);

  if (!items?.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = "No provider routing history is available yet.";
    providerMix.appendChild(empty);
    return;
  }

  items.forEach((item) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = `${toTitleCase(item.provider)}: ${item.assessments}`;
    providerMix.appendChild(chip);
  });
}

async function loadDashboard() {
  setLoadingState("Refreshing dashboard...");

  const response = await fetch("/api/doctor/dashboard");
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || "Failed to load doctor dashboard.");
  }

  renderMetrics(payload);
  renderRecentAssessments(payload.recent_assessments);
  renderPatientSummaries(payload.patient_summaries);
  renderProviderMix(payload.provider_distribution);
  doctorStatus.textContent = `Updated ${formatDateTime(payload.generated_at)} from persisted assessments.`;
}

refreshButton.addEventListener("click", () => {
  loadDashboard().catch((error) => {
    doctorStatus.textContent = error instanceof Error ? error.message : "Failed to refresh doctor dashboard.";
  });
});

loadDashboard().catch((error) => {
  doctorStatus.textContent = error instanceof Error ? error.message : "Failed to load doctor dashboard.";
});
