const messageList = document.querySelector("#message-list");
const composer = document.querySelector("#composer");
const promptInput = document.querySelector("#prompt-input");
const sendButton = document.querySelector("#send-button");
const connectionStatus = document.querySelector("#connection-status");
const coreState = document.querySelector("#core-state");
const modelSelector = document.querySelector("#model-selector");
const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8765" : "";
const MODEL_SELECTION_KEY = "lusas-selected-model";
let selectedModelId = localStorage.getItem(MODEL_SELECTION_KEY) || "sara-1.0";
const modelNames = { "sara-1.0": "Sara 1.0", "lira-1.0": "Lira 1.0" };

function setText(selector, value) {
  const element = document.querySelector(selector);
  if (element) element.textContent = value;
}

function setConnection(online) {
  connectionStatus.classList.toggle("offline", !online);
  connectionStatus.innerHTML = `<span class="status-dot"></span> ${online ? "CONNECTED" : "OFFLINE"}`;
  setText("#core-state", online ? "READY" : "OFFLINE");
}

function statusLabel(value) {
  return value ? "ON" : "OFF";
}

function updateStatus(payload) {
  const policy = payload.policy || {};
  const web = payload.web_learning || {};
  const knowledge = payload.knowledge || {};
  setText("#metric-version", payload.current_version || "—");
  setText("#metric-knowledge", knowledge.total ?? "—");
  setText("#metric-web", web.cached_articles ?? "—");
  setText("#metric-upgrades", payload.upgrade_count ?? "—");
  setText("#policy-model", statusLabel(policy.auto_apply_upgrades));
  setText("#policy-code", statusLabel(policy.auto_code_upgrades));
  setText("#policy-web", statusLabel(web.enabled));
  updateModels(payload.models, payload.selected_model_id);
}

function updateModels(models, serverSelected) {
  const entries = Array.isArray(models) ? models : [];
  if (serverSelected && modelNames[serverSelected]) selectedModelId = serverSelected;
  if (!modelNames[selectedModelId]) selectedModelId = "sara-1.0";
  if (!modelSelector) return;
  modelSelector.innerHTML = "";
  entries.forEach((model) => {
    const modelId = model.model_id || model.id;
    const option = document.createElement("option");
    option.value = modelId;
    option.textContent = model.display_name || modelNames[modelId] || modelId;
    modelSelector.appendChild(option);
  });
  if (!entries.length) {
    Object.entries(modelNames).forEach(([id, name]) => {
      const option = document.createElement("option");
      option.value = id;
      option.textContent = name;
      modelSelector.appendChild(option);
    });
  }
  modelSelector.value = selectedModelId;
  localStorage.setItem(MODEL_SELECTION_KEY, selectedModelId);
}

async function selectModel(modelId) {
  if (!modelNames[modelId]) return;
  const previous = selectedModelId;
  selectedModelId = modelId;
  localStorage.setItem(MODEL_SELECTION_KEY, modelId);
  try {
    const response = await fetch(`${API_BASE}/api/models/select`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Model selection failed.");
    selectedModelId = payload.selected_model_id || modelId;
    localStorage.setItem(MODEL_SELECTION_KEY, selectedModelId);
    modelSelector.value = selectedModelId;
    setText("#metric-version", modelNames[selectedModelId] || selectedModelId);
  } catch (error) {
    selectedModelId = previous;
    localStorage.setItem(MODEL_SELECTION_KEY, previous);
    modelSelector.value = previous;
    addMessage("assistant", `Model selection failed: ${error.message}`);
  }
}

async function refreshStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/status`, { cache: "no-store" });
    if (!response.ok) throw new Error("Status request failed");
    updateStatus(await response.json());
    setConnection(true);
  } catch (error) {
    setConnection(false);
  }
}

function addMessage(role, text, pending = false) {
  const message = document.createElement("article");
  message.className = `message ${role}-message${pending ? " pending" : ""}`;
  const label = role === "assistant" ? "LUSAS AI" : "YOU";
  const badge = role === "assistant" ? "L" : "Y";
  const type = role === "assistant" ? `${modelNames[selectedModelId] || "LOCAL MODEL"} · LOCAL MODEL` : "REQUEST";
  message.innerHTML = `
    <div class="avatar ${role === "assistant" ? "assistant-avatar" : "user-avatar"}">${badge}</div>
    <div class="message-body">
      <div class="message-meta"><strong>${label}</strong><span>${type}</span></div>
      <p></p>
    </div>`;
  message.querySelector("p").textContent = text;
  messageList.appendChild(message);
  message.scrollIntoView({ behavior: "smooth", block: "end" });
  return message;
}

function removeWelcomePrompts() {
  const prompts = document.querySelector("#prompt-grid");
  if (prompts) prompts.remove();
}

function setBusy(busy) {
  sendButton.disabled = busy;
  promptInput.disabled = busy;
  sendButton.querySelector("span").textContent = busy ? "Thinking" : "Send";
}

async function sendMessage(rawPrompt) {
  const prompt = rawPrompt.trim();
  if (!prompt || sendButton.disabled) return;
  removeWelcomePrompts();
  promptInput.value = "";
  promptInput.style.height = "auto";
  addMessage("user", prompt);
  const pending = addMessage("assistant", "Thinking…", true);
  setBusy(true);
  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, model_id: selectedModelId }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The local model could not answer.");
    pending.classList.remove("pending");
    pending.querySelector("p").textContent = payload.response;
    const modelLabel = modelNames[payload.model_id] || modelNames[selectedModelId] || "LOCAL MODEL";
    const meta = pending.querySelector(".message-meta span");
    if (meta) meta.textContent = `${modelLabel} · LOCAL MODEL`;
    setConnection(true);
  } catch (error) {
    pending.classList.remove("pending");
    pending.querySelector("p").textContent = error.message;
    pending.querySelector("p").style.color = "#f2a0a8";
  } finally {
    setBusy(false);
    promptInput.focus();
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(promptInput.value);
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

promptInput.addEventListener("input", () => {
  promptInput.style.height = "auto";
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 150)}px`;
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

document.querySelector("#clear-button").addEventListener("click", () => {
  messageList.innerHTML = "";
  addMessage("assistant", "Chat cleared. What would you like to explore?");
  promptInput.focus();
});

document.querySelector("#refresh-button").addEventListener("click", refreshStatus);
if (modelSelector) modelSelector.addEventListener("change", () => selectModel(modelSelector.value));
refreshStatus();
