const defaults = {
  model: "kraken-infer-qwen3-0.6b",
  device: "cpu",
  maxNewTokens: 16
};

const modelInput = document.getElementById("model");
const deviceInput = document.getElementById("device");
const tokensInput = document.getElementById("tokens");
const temperatureInput = document.getElementById("temperature");
const streamInput = document.getElementById("stream");
const thinkingInput = document.getElementById("thinking");
const messagesEl = document.getElementById("messages");
const emptyEl = document.getElementById("empty");
const form = document.getElementById("composer");
const promptInput = document.getElementById("prompt");
const sendButton = document.getElementById("send");
const stopButton = document.getElementById("stop");
const statusEl = document.getElementById("status");

const messages = [];
let controller = null;

function normalizeDevice(device) {
  return device === "mps" ? "mps:0" : device;
}

function hasDeviceOption(device) {
  return Array.from(deviceInput.options).some((option) => option.value === device);
}

function applyConfig(config) {
  const device = normalizeDevice(config.device || defaults.device);
  modelInput.value = config.model || defaults.model;
  tokensInput.value = String(config.max_new_tokens || defaults.maxNewTokens);
  if (hasDeviceOption(device)) {
    deviceInput.value = device;
  }
}

async function loadConfig() {
  applyConfig(defaults);
  try {
    const response = await fetch("/chat_page/config", {cache: "no-store"});
    if (!response.ok) {
      return;
    }
    applyConfig(await response.json());
  } catch (_) {
  }
}

function setStatus(value) {
  statusEl.textContent = value;
}

function setBusy(value) {
  sendButton.disabled = value;
  stopButton.disabled = !value;
  promptInput.disabled = value;
  setStatus(value ? "Busy" : "Ready");
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addBubble(role, text) {
  emptyEl.hidden = true;
  const node = document.createElement("div");
  node.className = "message " + role;
  node.textContent = text;
  messagesEl.appendChild(node);
  scrollToBottom();
  return node;
}

function requestBody(history) {
  const maxTokens = Math.max(1, Number.parseInt(tokensInput.value || "1", 10));
  const temperature = Number.parseFloat(temperatureInput.value);
  const body = {
    model: modelInput.value.trim() || defaults.model,
    messages: history,
    max_completion_tokens: maxTokens,
    stream: streamInput.checked,
    enable_thinking: thinkingInput.checked,
    device: deviceInput.value
  };
  if (Number.isFinite(temperature)) {
    body.temperature = temperature;
  }
  return body;
}

function contentFromPayload(payload) {
  return payload && payload.choices && payload.choices[0] &&
    payload.choices[0].message && payload.choices[0].message.content || "";
}

async function readStream(response, bubble) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) {
      break;
    }
    buffer += decoder.decode(chunk.value, {stream: true});
    let split = buffer.indexOf("\n\n");
    while (split !== -1) {
      const eventText = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      for (const rawLine of eventText.split("\n")) {
        const line = rawLine.trim();
        if (!line.startsWith("data:")) {
          continue;
        }
        const data = line.slice(5).trim();
        if (data === "[DONE]") {
          return text;
        }
        const payload = JSON.parse(data);
        const delta = payload && payload.choices && payload.choices[0] &&
          payload.choices[0].delta && payload.choices[0].delta.content || "";
        if (delta) {
          text += delta;
          bubble.textContent = text;
          scrollToBottom();
        }
      }
      split = buffer.indexOf("\n\n");
    }
  }
  return text;
}

async function sendMessage(text) {
  const nextMessages = messages.concat([{role: "user", content: text}]);
  addBubble("user", text);
  const assistantBubble = addBubble("assistant", "");
  controller = new AbortController();
  setBusy(true);
  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(requestBody(nextMessages)),
      signal: controller.signal
    });
    if (!response.ok) {
      let errorText = "HTTP " + response.status;
      try {
        const payload = await response.json();
        errorText = payload.error && payload.error.message || errorText;
      } catch (_) {
      }
      throw new Error(errorText);
    }
    let answer = "";
    if (streamInput.checked) {
      answer = await readStream(response, assistantBubble);
    } else {
      const payload = await response.json();
      answer = contentFromPayload(payload);
      assistantBubble.textContent = answer;
    }
    messages.push({role: "user", content: text});
    messages.push({role: "assistant", content: answer});
    scrollToBottom();
  } catch (error) {
    assistantBubble.remove();
    addBubble("error", error.name === "AbortError" ? "Stopped" : error.message);
  } finally {
    controller = null;
    setBusy(false);
    promptInput.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = promptInput.value.trim();
  if (!text || controller) {
    return;
  }
  promptInput.value = "";
  void sendMessage(text);
});

stopButton.addEventListener("click", () => {
  if (controller) {
    controller.abort();
  }
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

void loadConfig();
