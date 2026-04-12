const DEMO_LABELS = {
  "demo-dementia-case": "Demo: Dementia Case",
  "demo-healthy-case": "Demo: Healthy Case",
};

const SVG_NS = "http://www.w3.org/2000/svg";

const patientSelect = document.getElementById("patient-select");
const refreshButton = document.getElementById("refresh-dashboard");
const careStatus = document.getElementById("care-status");
const riskLabel = document.getElementById("risk-label");
const careSummary = document.getElementById("care-summary");
const heroPatient = document.getElementById("hero-patient");
const lastUpdatedAt = document.getElementById("last-updated-at");
const sessionsCompleted = document.getElementById("sessions-completed");
const heroStatusLabel = document.getElementById("hero-status-label");
const riskScore = document.getElementById("risk-score");
const scoreRing = document.getElementById("score-ring");
const scoreSupport = document.getElementById("score-support");
const scoreBreakdown = document.getElementById("score-breakdown");
const heroReasons = document.getElementById("hero-reasons");
const desktopTrendCard = document.getElementById("desktop-trend-card");
const mobileTrendSlot = document.getElementById("mobile-trend-slot");
const statusLabel = document.getElementById("status-label");
const reasonSummary = document.getElementById("reason-summary");
const baselineComparison = document.getElementById("baseline-comparison");
const longitudinalDirection = document.getElementById("longitudinal-direction");
const chartXAxisLabel = document.getElementById("chart-x-axis-label");
const chartYAxisLabel = document.getElementById("chart-y-axis-label");
const topReasons = document.getElementById("top-reasons");
const recommendation = document.getElementById("recommendation");
const nextSteps = document.getElementById("next-steps");
const alerts = document.getElementById("alerts");
const monitoringNotes = document.getElementById("monitoring-notes");
const historyList = document.getElementById("history-list");
const trendChart = document.getElementById("trend-chart");
const trendLabels = document.getElementById("trend-labels");

function formatDateTime(value) {
  if (!value) {
    return "-";
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString();
}

function formatScore(value) {
  return value === null || value === undefined || Number.isNaN(value) ? "-" : Number(value).toFixed(2);
}

function formatMetricValue(value) {
  return value === null || value === undefined || Number.isNaN(value) ? "-" : `${Math.round(Number(value) * 100)}%`;
}

function clearNode(node) {
  node.replaceChildren();
}

function displayPatientLabel(patientId) {
  return DEMO_LABELS[patientId] || patientId;
}

function statusPillClass(signal) {
  if (signal === "urgent") {
    return "status-pill status-pill-urgent";
  }
  if (signal === "watch") {
    return "status-pill status-pill-watch";
  }
  return "status-pill status-pill-stable";
}

function riskLabelClass(label) {
  return label === "HIGH RISK" ? "risk-label-high" : "risk-label-low";
}

function directionPillClass(direction) {
  if (direction === "declining") {
    return "trend-pill trend-pill-declining";
  }
  if (direction === "improving") {
    return "trend-pill trend-pill-improving";
  }
  return "trend-pill trend-pill-stable";
}

function chartPalette(direction) {
  if (direction === "declining") {
    return {
      line: "#d5432f",
      fill: "rgba(213, 67, 47, 0.12)",
    };
  }
  if (direction === "improving") {
    return {
      line: "#19775c",
      fill: "rgba(25, 119, 92, 0.12)",
    };
  }
  return {
    line: "#2850d8",
    fill: "rgba(40, 80, 216, 0.12)",
  };
}

function renderList(node, items, emptyMessage) {
  clearNode(node);

  if (!items?.length) {
    const item = document.createElement("li");
    item.className = "empty-state";
    item.textContent = emptyMessage;
    node.appendChild(item);
    return;
  }

  items.forEach((entry) => {
    const item = document.createElement("li");
    item.textContent = entry;
    node.appendChild(item);
  });
}

function renderReasonTags(items) {
  clearNode(heroReasons);

  if (!items?.length) {
    return;
  }

  items.slice(0, 3).forEach((entry) => {
    const item = document.createElement("li");
    item.className = "reason-chip";
    item.textContent = entry;
    heroReasons.appendChild(item);
  });
}

function renderScoreBreakdown(items) {
  clearNode(scoreBreakdown);

  if (!items?.length) {
    return;
  }

  items.forEach((entry) => {
    const item = document.createElement("li");
    item.className = "score-breakdown-item";

    const head = document.createElement("div");
    head.className = "score-breakdown-head";

    const label = document.createElement("span");
    label.className = "score-breakdown-label";
    label.textContent = entry.label || entry.key || "Metric";

    const value = document.createElement("strong");
    value.className = "score-breakdown-value";
    value.textContent = formatMetricValue(entry.value);

    head.appendChild(label);
    head.appendChild(value);

    const track = document.createElement("div");
    track.className = "score-breakdown-track";

    const fill = document.createElement("span");
    fill.className = "score-breakdown-fill";
    fill.style.setProperty("--metric-fill", `${Math.max(0, Math.min(1, Number(entry.value || 0))) * 100}%`);
    track.appendChild(fill);

    item.appendChild(head);
    item.appendChild(track);
    scoreBreakdown.appendChild(item);
  });
}

function renderHistory(items) {
  clearNode(historyList);

  if (!items?.length) {
    const item = document.createElement("li");
    item.className = "empty-state";
    item.textContent = "No shared history is available yet.";
    historyList.appendChild(item);
    return;
  }

  items.forEach((entry) => {
    const item = document.createElement("li");
    item.className = "history-item";

    const meta = document.createElement("div");
    meta.className = "history-meta";

    const date = document.createElement("strong");
    date.className = "history-date";
    date.textContent = formatDateTime(entry.created_at);

    const status = document.createElement("span");
    status.className = statusPillClass(entry.signal);
    status.textContent = entry.status_label || "Update";

    meta.appendChild(date);
    meta.appendChild(status);

    const summary = document.createElement("p");
    summary.className = "support-copy";
    summary.textContent = entry.shareable_summary || "No summary shared for this update.";

    item.appendChild(meta);
    item.appendChild(summary);
    historyList.appendChild(item);
  });
}

function buildReasonSummary(payload) {
  const reasons = Array.isArray(payload.top_reasons) ? payload.top_reasons.slice(0, 2) : [];
  if (!reasons.length) {
    return payload.care_summary || "No plain-language explanation is available yet.";
  }
  return `${payload.risk_label || "Current result"} is being driven mainly by ${reasons.join(" and ").toLowerCase()}.`;
}

function buildScoreSupport(payload) {
  if (payload.longitudinal_direction === "declining") {
    return "Trend is moving away from baseline and should be watched closely.";
  }
  if (payload.longitudinal_direction === "improving") {
    return "Trend is moving back toward baseline compared with previous checks.";
  }
  if (payload.risk_label === "HIGH RISK") {
    return "This result needs caregiver attention and clinician follow-up planning.";
  }
  return "This result stays lower risk, with routine monitoring still recommended.";
}

function updateUrl(patientId) {
  const url = new URL(window.location.href);
  url.searchParams.set("patient_id", patientId);
  window.history.replaceState({}, "", url);
}

function createSvgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => {
    element.setAttribute(key, String(value));
  });
  return element;
}

function renderTrendChart(points, direction) {
  clearNode(trendChart);
  clearNode(trendLabels);

  if (!points?.length) {
    const empty = createSvgElement("text", {
      x: 36,
      y: 110,
      fill: "#66748c",
      "font-size": "14",
    });
    empty.textContent = "No trend data yet.";
    trendChart.appendChild(empty);
    return;
  }

  const width = 320;
  const height = 200;
  const padding = { top: 22, right: 18, bottom: 42, left: 40 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const palette = chartPalette(direction);
  const lineColor = palette.line;
  const fillColor = palette.fill;
  const gridColor = "rgba(25, 35, 58, 0.10)";
  const axisColor = "rgba(25, 35, 58, 0.22)";

  trendChart.appendChild(
    createSvgElement("line", {
      x1: padding.left,
      y1: padding.top,
      x2: padding.left,
      y2: padding.top + plotHeight,
      stroke: axisColor,
      "stroke-width": 1.25,
    }),
  );
  trendChart.appendChild(
    createSvgElement("line", {
      x1: padding.left,
      y1: padding.top + plotHeight,
      x2: width - padding.right,
      y2: padding.top + plotHeight,
      stroke: axisColor,
      "stroke-width": 1.25,
    }),
  );

  [0, 0.5, 1].forEach((step) => {
    const y = padding.top + plotHeight - (plotHeight * step);
    trendChart.appendChild(
      createSvgElement("line", {
        x1: padding.left,
        y1: y,
        x2: width - padding.right,
        y2: y,
        stroke: gridColor,
        "stroke-width": 1,
      }),
    );

    const tickLabel = createSvgElement("text", {
      x: padding.left - 8,
      y: y + 4,
      fill: "#66748c",
      "font-size": "11",
      "text-anchor": "end",
    });
    tickLabel.textContent = step.toFixed(2);
    trendChart.appendChild(tickLabel);
  });

  const svgPoints = points.map((point, index) => {
    const x = padding.left + (plotWidth * index) / Math.max(1, points.length - 1);
    const y = padding.top + plotHeight - (plotHeight * Number(point.risk_score || 0));
    return { ...point, x, y };
  });

  const linePath = svgPoints
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(" ");
  const areaPath = [
    linePath,
    `L ${svgPoints[svgPoints.length - 1].x.toFixed(2)} ${padding.top + plotHeight}`,
    `L ${svgPoints[0].x.toFixed(2)} ${padding.top + plotHeight}`,
    "Z",
  ].join(" ");

  trendChart.appendChild(
    createSvgElement("path", {
      d: areaPath,
      fill: fillColor,
    }),
  );
  trendChart.appendChild(
    createSvgElement("path", {
      d: linePath,
      fill: "none",
      stroke: lineColor,
      "stroke-width": 4,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
    }),
  );

  svgPoints.forEach((point) => {
    trendChart.appendChild(
      createSvgElement("circle", {
        cx: point.x,
        cy: point.y,
        r: point.is_baseline ? 6 : 5,
        fill: "#ffffff",
        stroke: lineColor,
        "stroke-width": point.is_baseline ? 4 : 3,
      }),
    );
  });

  svgPoints.forEach((point) => {
    const tick = createSvgElement("line", {
      x1: point.x,
      y1: padding.top + plotHeight,
      x2: point.x,
      y2: padding.top + plotHeight + 6,
      stroke: axisColor,
      "stroke-width": 1,
    });
    trendChart.appendChild(tick);
  });

  points.forEach((point) => {
    const item = document.createElement("div");
    item.className = "chart-label";

    const labelNode = document.createElement("span");
    labelNode.textContent = point.label;

    const valueNode = document.createElement("strong");
    valueNode.textContent = formatScore(point.risk_score);

    item.appendChild(labelNode);
    item.appendChild(valueNode);
    trendLabels.appendChild(item);
  });
}

function syncMobileTrendCard() {
  if (!desktopTrendCard || !mobileTrendSlot) {
    return;
  }

  const clone = desktopTrendCard.cloneNode(true);
  clone.removeAttribute("id");
  clone.classList.remove("hero-chart-card");
  clone.classList.add("mobile-trend-card");
  clone.querySelectorAll("[id]").forEach((node) => node.removeAttribute("id"));
  mobileTrendSlot.replaceChildren(clone);
}

async function loadPatients() {
  const response = await fetch("/api/care/patients");
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || "Failed to load patient list.");
  }

  clearNode(patientSelect);
  const patients = Array.isArray(payload.patients) ? payload.patients : [];
  const options = patients.length ? patients : ["demo-dementia-case", "demo-healthy-case"];

  options.forEach((patientId) => {
    const option = document.createElement("option");
    option.value = patientId;
    option.textContent = displayPatientLabel(patientId);
    patientSelect.appendChild(option);
  });

  return options;
}

async function loadDashboard(patientId) {
  careStatus.textContent = "Refreshing care dashboard...";

  const response = await fetch(`/api/care/dashboard/${encodeURIComponent(patientId)}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || "Failed to load care dashboard.");
  }

  riskLabel.textContent = payload.risk_label || "-";
  riskLabel.className = riskLabelClass(payload.risk_label || "");
  careSummary.textContent = payload.care_summary || "-";
  heroPatient.textContent = displayPatientLabel(patientId);
  heroStatusLabel.textContent = payload.status_label || "-";
  lastUpdatedAt.textContent = formatDateTime(payload.last_updated_at);
  sessionsCompleted.textContent = payload.sessions_completed ?? "-";
  riskScore.textContent = formatScore(payload.risk_score);
  document.body.dataset.riskState = payload.risk_label === "HIGH RISK" ? "high" : "low";
  scoreRing.style.setProperty("--score-angle", `${Math.max(0, Math.min(1, Number(payload.risk_score || 0))) * 360}deg`);
  scoreRing.style.setProperty(
    "--score-color",
    payload.risk_label === "HIGH RISK" ? "#d5432f" : "#19775c",
  );
  scoreSupport.textContent = buildScoreSupport(payload);
  renderScoreBreakdown(payload.score_breakdown);
  reasonSummary.textContent = buildReasonSummary(payload);
  renderReasonTags(payload.top_reasons);

  statusLabel.textContent = payload.status_label || "-";
  statusLabel.className = statusPillClass(payload.signal);
  recommendation.textContent = payload.recommendation || "-";
  baselineComparison.textContent = payload.baseline_comparison || "-";
  longitudinalDirection.textContent = payload.longitudinal_direction_label || "-";
  longitudinalDirection.className = directionPillClass(payload.longitudinal_direction);
  chartXAxisLabel.textContent = `X-axis: ${payload.chart_x_axis_label || "Date"}`;
  chartYAxisLabel.textContent = `Y-axis: ${payload.chart_y_axis_label || "Risk Score"}`;
  renderList(topReasons, payload.top_reasons, "No clear drivers are available yet.");
  renderList(nextSteps, payload.next_steps, "No follow-up step is available yet.");
  renderList(alerts, payload.alerts, "No extra caregiver notes are active right now.");
  renderList(
    monitoringNotes,
    [payload.anomaly_summary, payload.identity_gating_summary].filter(Boolean),
    "No extra longitudinal monitoring notes are active right now.",
  );
  renderTrendChart(payload.longitudinal_points, payload.longitudinal_direction);
  syncMobileTrendCard();
  renderHistory(payload.history);

  careStatus.textContent = `Showing ${displayPatientLabel(patientId)}. Updated ${formatDateTime(payload.generated_at)}.`;
  updateUrl(patientId);
}

async function initialize() {
  const url = new URL(window.location.href);
  const requestedPatientId = url.searchParams.get("patient_id");
  const patients = await loadPatients();
  const initialPatientId = requestedPatientId && patients.includes(requestedPatientId)
    ? requestedPatientId
    : patients[0];

  patientSelect.value = initialPatientId;
  await loadDashboard(initialPatientId);
}

patientSelect.addEventListener("change", () => {
  loadDashboard(patientSelect.value).catch((error) => {
    careStatus.textContent = error instanceof Error ? error.message : "Failed to load patient dashboard.";
  });
});

refreshButton.addEventListener("click", () => {
  loadDashboard(patientSelect.value).catch((error) => {
    careStatus.textContent = error instanceof Error ? error.message : "Failed to refresh patient dashboard.";
  });
});

initialize().catch((error) => {
  careStatus.textContent = error instanceof Error ? error.message : "Failed to initialize care dashboard.";
});
