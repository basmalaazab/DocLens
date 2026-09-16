// API_BASE is configured in config.js (gitignored).
// To change the backend URL, copy config.js.example → config.js and edit it.
// Falls back to localhost:8000 when config.js is absent (e.g. local dev without the file).
const API_BASE = (typeof window !== "undefined" && window.API_BASE)
  ? window.API_BASE
  : "http://127.0.0.1:8000";

const $ = (id) => document.getElementById(id);
const result = $("result");
const loading = $("loading");
const status = $("status");
const imageInput = $("image-input");
const previewWrap = $("preview-wrap");
const preview = $("preview");
const dropzone = $("dropzone");
const apiBadge = $("api-badge");
const assistantForm = $("assistant-form");

let loadingTimer = null;
function setLoading(on) {
  loading.classList.toggle("hidden", !on);
  const textEl = loading.querySelector("span");
  if (on) {
    let sec = 0;
    if (textEl) textEl.textContent = "Retrieving relevant document chunks...";
    clearInterval(loadingTimer);
    loadingTimer = setInterval(() => {
      sec += 2;
      if (sec >= 4 && sec < 8 && textEl) {
        textEl.textContent = "Analyzing context with Ollama (Llama 3.2)...";
      } else if (sec >= 8 && textEl) {
        textEl.textContent = `Generating grounded answer with citations... (${sec}s)`;
      }
    }, 2000);
  } else {
    clearInterval(loadingTimer);
  }
}

function showResult(data, isVision=false) {
  result.classList.remove("hidden");
  $("answer").textContent = data.answer || "No answer returned.";
  const sources = data.sources || [];
  $("source-count").textContent = sources.length;

  $("source-list").innerHTML = sources.length
    ? sources.map(s => {
        const text = typeof s === 'string' ? s : `${s.source} - Page ${s.page}`;
        return `<div class="source-item">${escapeHtml(text)}</div>`;
      }).join("")
    : `<div class="source-item">No sources returned.</div>`;

  $("vision-info").classList.toggle("hidden", !isVision);
  if (isVision) {
    $("content-type").textContent = data.content_type || "unknown";
    $("detected-elements").textContent = data.detected_elements || "No elements detected.";
  }
  result.scrollIntoView({behavior:"smooth", block:"start"});
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, c => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"
  }[c]));
}

// Update UI state when an image is attached or removed
imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) {
    previewWrap.classList.add("hidden");
    apiBadge.textContent = "POST /query";
    return;
  }
  preview.src = URL.createObjectURL(file);
  previewWrap.classList.remove("hidden");
  apiBadge.textContent = "POST /query-image";
});

$("remove-image").addEventListener("click", () => {
  imageInput.value = "";
  preview.src = "";
  previewWrap.classList.add("hidden");
  apiBadge.textContent = "POST /query";
});

// Single unified form submission
assistantForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("question").value.trim();
  const file = imageInput.files[0];

  if (!question) return alert("Please enter a question.");

  setLoading(true);
  result.classList.add("hidden");

  try {
    let res, data;
    if (file) {
      // Vision Query (Question + Image)
      const form = new FormData();
      form.append("question", question);
      form.append("image", file);
      res = await fetch(`${API_BASE}/query-image`, {method: "POST", body: form});
      data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Image query failed.");
      showResult(data, true);
    } else {
      // Text Query (Question only)
      res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question})
      });
      data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Query failed.");
      showResult(data, false);
    }
  } catch (err) {
    alert(`Could not connect to backend.\n\n${err.message}`);
  } finally {
    setLoading(false);
  }
});

["dragenter","dragover"].forEach(evt => dropzone.addEventListener(evt, e => {
  e.preventDefault(); dropzone.classList.add("drag");
}));
["dragleave","drop"].forEach(evt => dropzone.addEventListener(evt, e => {
  e.preventDefault(); dropzone.classList.remove("drag");
}));
dropzone.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  if (!file || !file.type.startsWith("image/")) return;
  const dt = new DataTransfer();
  dt.items.add(file);
  imageInput.files = dt.files;
  imageInput.dispatchEvent(new Event("change"));
});

$("copy-btn").addEventListener("click", async () => {
  await navigator.clipboard.writeText($("answer").textContent);
  $("copy-btn").textContent = "Copied ✓";
  setTimeout(() => $("copy-btn").textContent = "Copy answer", 1300);
});

async function checkBackend() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error();
    const data = await res.json();
    status.classList.add("online");
    status.innerHTML = `<span class="status-dot"></span> Backend online · ${data.collection_count} chunks`;
  } catch (err) {
    status.classList.remove("online");
    status.innerHTML = `<span class="status-dot"></span> Backend offline`;
  }
}
checkBackend();
setInterval(checkBackend, 3000);
