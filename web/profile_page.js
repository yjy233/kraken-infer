const statusEl = document.getElementById("status");
const rootPathEl = document.getElementById("rootPath");
const requestListEl = document.getElementById("requestList");
const requestIdEl = document.getElementById("requestId");
const requestMetaEl = document.getElementById("requestMeta");
const summaryEl = document.getElementById("summary");
const linksEl = document.getElementById("links");
const svgWrapEl = document.getElementById("svgWrap");

const params = new URLSearchParams(window.location.search);
let activeRequestId = params.get("request_id") || "";
let profileRoot = "build/profiles";

function setStatus(text, error = false) {
  statusEl.textContent = text;
  statusEl.className = error ? "status error" : "status";
}

function escapeText(value) {
  const node = document.createElement("div");
  node.textContent = value;
  return node.innerHTML;
}

function link(href, label) {
  return `<a href="${href}" target="_blank" rel="noreferrer">${escapeText(label)}</a>`;
}

function renderLinks(requestId) {
  linksEl.innerHTML = [
    link(`/profiles/${encodeURIComponent(requestId)}/summary.json`, "summary.json"),
    link(`/profiles/${encodeURIComponent(requestId)}/summary.txt`, "summary.txt"),
    link(`/profiles/${encodeURIComponent(requestId)}/trace.json`, "trace.json"),
    link(`/profiles/${encodeURIComponent(requestId)}/profile.folded`, "profile.folded"),
    link(`/profiles/${encodeURIComponent(requestId)}/profile.svg`, "profile.svg")
  ].join("");
}

function renderList(profiles) {
  requestListEl.innerHTML = "";
  if (!profiles.length) {
    requestListEl.innerHTML = '<div class="empty">No profile records</div>';
    return;
  }
  for (const profile of profiles) {
    const item = document.createElement("div");
    item.className = "item" + (profile.request_id === activeRequestId ? " active" : "");
    item.innerHTML = `
      <div class="id">${escapeText(profile.request_id)}</div>
      <div class="meta">${escapeText(profile.created_at || "")} ${escapeText(profile.device || "")} ${escapeText(profile.profile_mode || "")}</div>
      <div class="meta">${escapeText(profile.status || "")} ${escapeText(profile.model_dir || "")}</div>
    `;
    item.addEventListener("click", () => {
      activeRequestId = profile.request_id;
      void loadProfile(profile.request_id);
    });
    requestListEl.appendChild(item);
  }
}

function renderManifest(manifest) {
  requestIdEl.textContent = manifest.request_id || "Profile";
  requestMetaEl.textContent = `${manifest.created_at || ""} ${manifest.device || ""} ${manifest.profile_mode || ""}`.trim();
  renderLinks(manifest.request_id);
}

function renderSummary(text) {
  summaryEl.textContent = text || "No summary";
}

function renderSvg(svgText) {
  if (!svgText) {
    svgWrapEl.innerHTML = '<div class="empty">No flamegraph</div>';
    return;
  }
  svgWrapEl.innerHTML = svgText;
}

async function fetchJson(path) {
  const response = await fetch(path, {cache: "no-store"});
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

async function fetchText(path) {
  const response = await fetch(path, {cache: "no-store"});
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.text();
}

async function loadProfile(requestId) {
  setStatus(`Loading ${requestId}`);
  try {
    const manifest = await fetchJson(`/profiles/${encodeURIComponent(requestId)}/manifest.json`);
    const summary = await fetchText(`/profiles/${encodeURIComponent(requestId)}/summary.txt`);
    renderManifest(manifest);
    renderSummary(summary);
    try {
      const svg = await fetchText(`/profiles/${encodeURIComponent(requestId)}/profile.svg`);
      renderSvg(svg);
    } catch (_) {
      renderSvg("");
    }
    setStatus("Ready");
    activeRequestId = requestId;
    const items = Array.from(requestListEl.querySelectorAll(".item"));
    for (const item of items) {
      item.classList.toggle("active", item.querySelector(".id")?.textContent === requestId);
    }
  } catch (error) {
    requestIdEl.textContent = requestId;
    requestMetaEl.textContent = "";
    renderSummary(error.message);
    renderSvg("");
    setStatus(error.message, true);
  }
}

async function loadIndex() {
  try {
    const config = await fetchJson("/profile_page/config");
    profileRoot = config.profile_dir || profileRoot;
  } catch (_) {
  }
  rootPathEl.textContent = profileRoot;
  const payload = await fetchJson("/profiles/index.json");
  const profiles = payload.profiles || [];
  renderList(profiles);
  if (!activeRequestId && profiles.length) {
    activeRequestId = profiles[0].request_id;
  }
  if (activeRequestId) {
    await loadProfile(activeRequestId);
  } else {
    setStatus("Ready");
  }
}

void loadIndex().catch((error) => {
  setStatus(error.message, true);
  summaryEl.textContent = error.message;
});
