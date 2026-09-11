const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8765" : "";
const TOKEN_KEY = "lusas-admin-token";
const tokenInput = document.querySelector("#admin-token");
const statusLabel = document.querySelector("#admin-status");
const modelNames = { "sara-1.0": "Sara 1.0", "lira-1.0": "Lira 1.0" };
tokenInput.value = localStorage.getItem(TOKEN_KEY) || "";

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>\"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#039;" })[character]);
}

function setStatus(text, error = false) {
  statusLabel.textContent = text;
  statusLabel.classList.toggle("offline", error);
}

function authHeaders(json = false) {
  const headers = { "X-LUSAS-Admin-Token": localStorage.getItem(TOKEN_KEY) || "" };
  if (json) headers["Content-Type"] = "application/json";
  return headers;
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...options, headers: { ...authHeaders(Boolean(options.body)), ...(options.headers || {}) } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}

function renderModels(models) {
  const target = document.querySelector("#model-cards");
  target.innerHTML = models.map((model) => `<article class="model-card"><div class="model-card-head"><div><div class="eyebrow">${escapeHtml(model.display_name)}</div><h3>${escapeHtml(model.model_id)}</h3></div><span class="state-pill ${model.stable_available ? "good" : "warn"}">${model.stable_available ? "STABLE" : "NOT INSTALLED"}</span></div><dl><div><dt>Foundation</dt><dd>${escapeHtml(model.foundation_model || model.foundation_path || "Not configured")}</dd></div><div><dt>Stable version</dt><dd>${escapeHtml(model.stable_version || "—")}</dd></div><div><dt>Candidates</dt><dd>${escapeHtml(model.candidate_count)}</dd></div></dl><button class="clear-button train-button" data-model-id="${escapeHtml(model.model_id)}" type="button">Start gated training</button></article>`).join("");
  target.querySelectorAll(".train-button").forEach((button) => button.addEventListener("click", () => startTraining(button.dataset.modelId)));
}

function renderCandidates(candidates) {
  const target = document.querySelector("#candidate-rows");
  if (!candidates.length) { target.innerHTML = '<tr><td colspan="5">No candidate evaluation reports.</td></tr>'; return; }
  target.innerHTML = candidates.map((candidate) => { const gates = candidate.gates || {}; const gateText = Object.keys(gates).length ? Object.entries(gates).map(([name, passed]) => `${name}:${passed ? "pass" : "fail"}`).join(" · ") : "not evaluated"; const metrics = candidate.metrics || {}; return `<tr><td>${escapeHtml(candidate.display_name)}</td><td><span class="table-muted">${escapeHtml(candidate.candidate_path)}</span></td><td>${escapeHtml(candidate.score ?? "—")} · ${candidate.passed ? "PASS" : "FAIL"}</td><td>${escapeHtml(gateText)}</td><td>${escapeHtml(metrics.max_latency_seconds ?? "—")}s / ${escapeHtml(metrics.resource_rss_mb ?? "—")}MB</td></tr>`; }).join("");
}

function renderKnowledge(records) {
  const target = document.querySelector("#knowledge-rows");
  if (!records.length) { target.innerHTML = '<tr><td colspan="5">No knowledge records found.</td></tr>'; return; }
  target.innerHTML = records.slice(0, 100).map((record) => { const id = record.id || record.knowledge_id || ""; const findings = (record.malicious_findings || []).join(", ") || "none"; return `<tr><td><strong>${escapeHtml(id)}</strong><br><span class="table-muted">${escapeHtml((record.text || record.content || "").slice(0, 180))}</span></td><td>${escapeHtml(record.category || record.domain || "general")}</td><td>${escapeHtml(record.source || "local")}<br><span class="table-muted">scan: ${escapeHtml(findings)}</span></td><td>${escapeHtml(record.approval_status || "pending")}</td><td><button class="tiny-button review-button" data-id="${escapeHtml(id)}" data-status="approved" type="button">Approve</button><button class="tiny-button reject review-button" data-id="${escapeHtml(id)}" data-status="rejected" type="button">Reject</button></td></tr>`; }).join("");
  target.querySelectorAll(".review-button").forEach((button) => button.addEventListener("click", () => reviewKnowledge(button.dataset.id, button.dataset.status)));
}

function renderSources(sources) {
  const target = document.querySelector("#approved-sources");
  target.innerHTML = sources.length ? `<span class="table-muted">Approved HTTPS sources: ${sources.map((source) => escapeHtml(source.url)).join(" · ")}</span>` : "<span class=\"table-muted\">No approved sources.</span>";
}

function renderJobs(jobs) {
  const target = document.querySelector("#training-jobs");
  target.innerHTML = jobs.length ? jobs.map((job) => `<div class="stack-item"><strong>${escapeHtml(modelNames[job.model_id] || job.model_id)}</strong><span>${escapeHtml(job.status)} · ${escapeHtml(job.dataset_count)} approved records</span><small>${escapeHtml(job.error || job.created_at)}</small></div>`).join("") : "<p>No training jobs yet.</p>";
}

function renderAudit(events) {
  const target = document.querySelector("#audit-events");
  target.innerHTML = events.length ? events.slice(0, 30).map((event) => `<div class="stack-item"><strong>${escapeHtml(event.action)}</strong><span>${escapeHtml(event.subject)} · ${escapeHtml(event.actor)}</span><small>${escapeHtml(event.created_at)}</small></div>`).join("") : "<p>No audit events yet.</p>";
}

async function loadAdmin() {
  try {
    const [overview, knowledge, audit] = await Promise.all([request("/api/admin/overview"), request("/api/admin/knowledge"), request("/api/admin/audit")]);
    renderModels(overview.models || []); renderCandidates(overview.candidates || []); renderKnowledge(knowledge.records || []); renderSources(overview.sources || []); renderJobs(overview.training_jobs || []); renderAudit(audit.events || []); setStatus("ADMIN CONNECTED");
  } catch (error) { setStatus(error.message, true); }
}

async function startTraining(modelId) {
  try { await request("/api/admin/training/start", { method: "POST", body: JSON.stringify({ model_id: modelId }) }); setStatus(`TRAINING QUEUED · ${modelNames[modelId]}`); await loadAdmin(); } catch (error) { setStatus(error.message, true); }
}

async function reviewKnowledge(recordId, status) {
  try { await request("/api/admin/knowledge/review", { method: "POST", body: JSON.stringify({ record_id: recordId, status }) }); await loadAdmin(); } catch (error) { setStatus(error.message, true); }
}

function evaluationPayload() { return JSON.parse(document.querySelector("#evaluation-json").value); }

document.querySelector("#save-token").addEventListener("click", () => { localStorage.setItem(TOKEN_KEY, tokenInput.value.trim()); loadAdmin(); });
document.querySelector("#refresh-admin").addEventListener("click", loadAdmin);
document.querySelector("#approve-source").addEventListener("click", async () => { try { await request("/api/admin/sources", { method: "POST", body: JSON.stringify({ url: document.querySelector("#source-url").value.trim() }) }); document.querySelector("#source-url").value = ""; setStatus("SOURCE APPROVED"); await loadAdmin(); } catch (error) { setStatus(error.message, true); } });
document.querySelector("#approve-deployment").addEventListener("click", async () => { try { await request("/api/admin/deployments/approve", { method: "POST", body: JSON.stringify({ model_id: document.querySelector("#deployment-model").value, candidate_path: document.querySelector("#candidate-path").value, evaluation: evaluationPayload() }) }); setStatus("DEPLOYMENT APPROVED"); await loadAdmin(); } catch (error) { setStatus(error.message, true); } });
document.querySelector("#deploy-approved").addEventListener("click", async () => { try { await request("/api/admin/deployments/deploy", { method: "POST", body: JSON.stringify({ model_id: document.querySelector("#deployment-model").value, candidate_path: document.querySelector("#candidate-path").value, evaluation: evaluationPayload() }) }); setStatus("DEPLOYED TO STABLE"); await loadAdmin(); } catch (error) { setStatus(error.message, true); } });
document.querySelector("#rollback-form").addEventListener("submit", async (event) => { event.preventDefault(); try { await request("/api/admin/deployments/rollback", { method: "POST", body: JSON.stringify({ model_id: document.querySelector("#deployment-model").value, backup_path: document.querySelector("#backup-path").value }) }); setStatus("ROLLED BACK TO BACKUP"); await loadAdmin(); } catch (error) { setStatus(error.message, true); } });
loadAdmin();
