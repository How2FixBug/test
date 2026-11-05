const OLLAMA_URL = "http://localhost:11434";

const form = document.querySelector("#chat-form");
const promptInput = document.querySelector("#prompt");
const sendButton = document.querySelector("#send");
const modelSelect = document.querySelector("#model");
const messages = document.querySelector("#messages");
const welcome = document.querySelector("#welcome");
const connection = document.querySelector("#connection");
const connectionText = document.querySelector("#connection-text");

let busy = false;

function setConnection(online, label) {
  connection.classList.toggle("online", online);
  connection.classList.toggle("offline", !online);
  connectionText.textContent = label;
}

async function loadModels() {
  try {
    const response = await fetch(`${OLLAMA_URL}/api/tags`);
    if (!response.ok) throw new Error("Ollama did not respond");
    const data = await response.json();
    const models = data.models || [];
    modelSelect.replaceChildren();

    if (!models.length) {
      modelSelect.add(new Option("No models installed", ""));
      setConnection(true, "Ollama connected · no models");
      return;
    }

    for (const model of models) {
      modelSelect.add(new Option(model.name, model.name));
    }
    setConnection(true, "Ollama connected");
    sendButton.disabled = false;
  } catch {
    modelSelect.replaceChildren(new Option("Ollama unavailable", ""));
    setConnection(false, "Ollama not connected");
  }
}

function addMessage(role, content, extraClass = "") {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;

  const label = document.createElement("div");
  label.className = "message-role";
  label.textContent = role === "user" ? "You" : "Ultillama";

  const body = document.createElement("div");
  body.className = `message-content ${extraClass}`;
  body.textContent = content;

  wrapper.append(label, body);
  messages.append(wrapper);
  wrapper.scrollIntoView({ behavior: "smooth", block: "end" });
  return body;
}

async function sendMessage(text) {
  const model = modelSelect.value;
  if (!model || busy) return;

  busy = true;
  sendButton.disabled = true;
  addMessage("user", text);
  welcome.hidden = true;

  const answer = addMessage("assistant", "");
  answer.classList.add("typing");
  answer.textContent = "Thinking…";

  try {
    // Intentionally send ONLY this message. No prior turns, memory, or system context.
    const response = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: text }],
        stream: true
      })
    });

    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `Request failed (${response.status})`);
    }

    answer.classList.remove("typing");
    answer.textContent = "";

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.trim()) continue;
        const chunk = JSON.parse(line);
        if (chunk.message?.content) {
          answer.textContent += chunk.message.content;
          answer.parentElement.scrollIntoView({ behavior: "smooth", block: "end" });
        }
      }
    }

    if (buffer.trim()) {
      const chunk = JSON.parse(buffer);
      if (chunk.message?.content) answer.textContent += chunk.message.content;
    }
    if (!answer.textContent) answer.textContent = "(No response)";
  } catch (error) {
    answer.classList.remove("typing");
    answer.classList.add("error");
    answer.textContent = `Couldn't reach Ollama: ${error.message}. Check that Ollama is running and allows this page's origin.`;
  } finally {
    busy = false;
    sendButton.disabled = !modelSelect.value;
    promptInput.focus();
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = promptInput.value.trim();
  if (!text || busy || !modelSelect.value) return;
  promptInput.value = "";
  promptInput.style.height = "auto";
  await sendMessage(text);
});

promptInput.addEventListener("input", () => {
  promptInput.style.height = "auto";
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 180)}px`;
  sendButton.disabled = busy || !modelSelect.value || !promptInput.value.trim();
});

promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

modelSelect.addEventListener("change", () => {
  sendButton.disabled = busy || !modelSelect.value || !promptInput.value.trim();
});

loadModels();
setInterval(loadModels, 15000);