const OLLAMA_URL = "http://localhost:11434";
const STORAGE_KEY = "ultillama.data.v2";
const DEFAULTS = { contextSize: 10, memoryEnabled: true, autoMemory: true, temperature: 0.7, embeddingModel: "nomic-embed-text" };

const $ = (selector) => document.querySelector(selector);
const form = $("#chat-form");
const promptInput = $("#prompt");
const sendButton = $("#send");
const modelSelect = $("#model");
const messagesEl = $("#messages");
const welcome = $("#welcome");
const connection = $("#connection");
const connectionText = $("#connection-text");
const chatList = $("#chat-list");

let state = loadState();
let busy = false;
let abortController = null;

function uid() { return crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2); }
function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved && Array.isArray(saved.chats)) return {
      chats: saved.chats, activeChatId: saved.activeChatId || null,
      memories: Array.isArray(saved.memories) ? saved.memories : [],
      memoryVectors: saved.memoryVectors && typeof saved.memoryVectors === "object" ? saved.memoryVectors : {},
      settings: { ...DEFAULTS, ...(saved.settings || {}) }
    };
  } catch (error) { console.warn("Could not load saved data", error); }
  return { chats: [], activeChatId: null, memories: [], memoryVectors: {}, settings: { ...DEFAULTS } };
}
function persist() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }
  catch { setConnection(false, "Storage full — delete old chats"); }
}
function activeChat() { return state.chats.find(chat => chat.id === state.activeChatId) || null; }
function setConnection(online, label) {
  connection.classList.toggle("online", online);
  connection.classList.toggle("offline", !online);
  connectionText.textContent = label;
}
function makeChat() {
  const chat = { id: uid(), title: "New chat", createdAt: Date.now(), updatedAt: Date.now(), messages: [] };
  state.chats.unshift(chat);
  state.activeChatId = chat.id;
  persist();
  renderAll();
  promptInput.focus();
}
function deleteChat(id) {
  state.chats = state.chats.filter(chat => chat.id !== id);
  if (state.activeChatId === id) state.activeChatId = state.chats[0]?.id || null;
  persist();
  renderAll();
}
function renderChatList() {
  chatList.replaceChildren();
  for (const chat of state.chats) {
    const row = document.createElement("div");
    row.className = "chat-row" + (chat.id === state.activeChatId ? " active" : "");
    const open = document.createElement("button");
    open.className = "chat-open";
    open.textContent = chat.title;
    open.title = chat.title;
    open.addEventListener("click", () => { if (busy) return; state.activeChatId = chat.id; persist(); renderAll(); $("#sidebar").classList.remove("mobile-open"); });
    open.addEventListener("dblclick", event => { event.preventDefault(); const title = window.prompt("Rename chat", chat.title); if (title && title.trim()) { chat.title = title.trim().slice(0, 60); persist(); renderAll(); } });
    const remove = document.createElement("button");
    remove.className = "chat-delete";
    remove.textContent = "×";
    remove.title = "Delete chat";
    remove.setAttribute("aria-label", "Delete " + chat.title);
    remove.addEventListener("click", () => { if (!busy && confirm("Delete this chat?")) deleteChat(chat.id); });
    row.append(open, remove);
    chatList.append(row);
  }
}
function renderMessages() {
  messagesEl.replaceChildren();
  const chat = activeChat();
  const hasMessages = !!chat?.messages.length;
  welcome.hidden = hasMessages;
  $("#active-title").textContent = chat?.title || "New chat";
  if (!chat) return;
  for (const message of chat.messages) {
    const wrapper = document.createElement("article");
    wrapper.className = "message " + message.role;
    const label = document.createElement("div");
    label.className = "message-role";
    label.textContent = message.role === "user" ? "You" : "Ultillama";
    const body = document.createElement("div");
    body.className = "message-content";
    body.textContent = message.content;
    wrapper.append(label, body);
    messagesEl.append(wrapper);
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}
function renderMemories() {
  const list = $("#memory-list");
  list.replaceChildren();
  if (!state.memories.length) {
    const empty = document.createElement("p");
    empty.className = "empty-memory";
    empty.textContent = "No saved memories yet. Add one above, or enable automatic memory in Settings.";
    list.append(empty);
    return;
  }
  state.memories.forEach((memory, index) => {
    const row = document.createElement("div");
    row.className = "memory-item";
    const content = document.createElement("p");
    content.textContent = memory;
    const edit = document.createElement("button");
    edit.textContent = "Edit";
    edit.addEventListener("click", () => {
      const updated = prompt("Edit memory", memory);
      if (updated === null) return;
      if (updated.trim()) state.memories[index] = updated.trim();
      else state.memories.splice(index, 1);
      persist(); renderMemories();
    });
    const remove = document.createElement("button");
    remove.textContent = "×";
    remove.title = "Delete memory";
    remove.addEventListener("click", () => { delete state.memoryVectors[memory]; state.memories.splice(index, 1); persist(); renderMemories(); });
    row.append(content, edit, remove);
    list.append(row);
  });
}
function renderAll() {
  renderChatList();
  renderMessages();
  renderMemories();
  $("#context-note").textContent = `Short-term context ${state.settings.contextSize ? "on" : "off"} · long-term memory ${state.settings.memoryEnabled ? "on" : "off"}`;
  $("#context-size").value = String(state.settings.contextSize);
  $("#memory-enabled").checked = state.settings.memoryEnabled;
  $("#auto-memory").checked = state.settings.autoMemory;
  $("#temperature").value = state.settings.temperature;
  $("#embedding-model").value = state.settings.embeddingModel || "nomic-embed-text";
}
function addMessage(role, content) {
  const chat = activeChat() || (makeChat(), activeChat());
  const message = { role, content, createdAt: Date.now() };
  chat.messages.push(message);
  chat.updatedAt = Date.now();
  if (role === "user" && chat.title === "New chat") {
    chat.title = content.trim().replace(/\s+/g, " ").slice(0, 40) || "New chat";
  }
  persist();
  renderAll();
  return message;
}
async function getEmbedding(text) {
  const model = (state.settings.embeddingModel || "nomic-embed-text").trim();
  if (!model) throw new Error("No embedding model selected");
  const response = await fetch(`${OLLAMA_URL}/api/embed`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, input: text })
  });
  if (!response.ok) throw new Error("Embedding model unavailable");
  const data = await response.json();
  if (!Array.isArray(data.embeddings?.[0])) throw new Error("Invalid embedding response");
  return data.embeddings[0];
}
function cosineSimilarity(a, b) {
  let dot = 0, aa = 0, bb = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) { dot += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  return dot / (Math.sqrt(aa) * Math.sqrt(bb) || 1);
}
async function retrieveRelevantMemories(query) {
  if (!state.settings.memoryEnabled || !state.memories.length) return [];
  try {
    const missing = state.memories.filter(memory => !Array.isArray(state.memoryVectors[memory]));
    if (missing.length) {
      const response = await fetch(`${OLLAMA_URL}/api/embed`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: state.settings.embeddingModel || "nomic-embed-text", input: missing })
      });
      if (!response.ok) throw new Error("Embedding model unavailable");
      const data = await response.json();
      if (!Array.isArray(data.embeddings) || data.embeddings.length !== missing.length) throw new Error("Invalid embeddings");
      missing.forEach((memory, index) => { state.memoryVectors[memory] = data.embeddings[index]; });
      persist();
    }
    const queryVector = await getEmbedding(query);
    return state.memories.map(text => ({
      text, score: cosineSimilarity(queryVector, state.memoryVectors[text] || [])
    })).sort((a, b) => b.score - a.score).slice(0, 6).filter(item => item.score > 0.18);
  } catch (error) {
    console.warn("Semantic memory search unavailable; using keyword matching.", error);
    const queryWords = new Set((query.toLowerCase().match(/[a-z0-9']{2,}/g) || []));
    return state.memories.map(text => {
      const memoryWords = new Set((text.toLowerCase().match(/[a-z0-9']{2,}/g) || []));
      let overlap = 0;
      for (const word of queryWords) if (memoryWords.has(word)) overlap++;
      return { text, score: overlap / Math.max(1, Math.sqrt(memoryWords.size)) };
    }).filter(item => item.score > 0.04).sort((a, b) => b.score - a.score).slice(0, 6);
  }
}
async function buildRequestMessages(chat) {
  const currentText = chat.messages.at(-1)?.content || "";
  const history = chat.messages.slice(0, -1);
  const recent = state.settings.contextSize === 0 ? [] : history.slice(-state.settings.contextSize);
  const selectedMemories = await retrieveRelevantMemories(currentText);
  const systemParts = [];
  if (selectedMemories.length) systemParts.push("Relevant long-term memories (retrieved by semantic similarity; use only when helpful):\\n" + selectedMemories.map(item => "- " + item.text).join("\\n"));
  if (systemParts.length) systemParts.push("Use recent conversation messages for short-term context. If a saved memory conflicts with what the user says now, trust what they say now.");
  return [
    ...(systemParts.length ? [{ role: "system", content: systemParts.join("\\n\\n") }] : []),
    ...recent.map(({ role, content }) => ({ role, content })),
    { role: "user", content: currentText }
  ];
}
async function updateLongTermMemory(userText, assistantText, model) {
  if (!state.settings.autoMemory) return;
  const prompt = [
    "You maintain a small, useful long-term memory for this user.",
    "Extract only stable preferences, ongoing projects, or facts the user clearly wants remembered.",
    "Do not save temporary details, guesses, secrets, sensitive personal information, or facts only about fictional roleplay.",
    "Return ONLY valid JSON in this exact shape: {\"add\":[\"short memory\"],\"remove\":[\"exact old memory to remove\"]}.",
    "If nothing should change, return {\"add\":[],\"remove\":[]}. Keep at most 30 memories. Each memory must be concise.",
    "USER MESSAGE: " + userText,
    "ASSISTANT REPLY: " + assistantText.slice(0, 1800),
    "EXISTING MEMORIES:\n" + state.memories.map(item => "- " + item).join("\n")
  ].join("\n\n");
  try {
    const response = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, stream: false, format: "json", messages: [{ role: "user", content: prompt }] })
    });
    if (!response.ok) return;
    const data = await response.json();
    const parsed = JSON.parse(data.message?.content || "{}");
    const remove = Array.isArray(parsed.remove) ? parsed.remove : [];
    const add = Array.isArray(parsed.add) ? parsed.add : [];
    state.memories = state.memories.filter(item => !remove.includes(item));
    for (const oldMemory of remove) delete state.memoryVectors[oldMemory];
    for (const item of add) {
      if (typeof item === "string" && item.trim() && !state.memories.includes(item.trim())) state.memories.push(item.trim().slice(0, 300));
    }
    state.memories = state.memories.slice(-30);
    for (const key of Object.keys(state.memoryVectors)) if (!state.memories.includes(key)) delete state.memoryVectors[key];
    persist(); renderMemories(); renderAll();
  } catch (error) { console.warn("Automatic memory update skipped", error); }
}
async function loadModels() {
  try {
    const response = await fetch(`${OLLAMA_URL}/api/tags`);
    if (!response.ok) throw new Error("Ollama did not respond");
    const data = await response.json();
    const models = data.models || [];
    const previous = modelSelect.value;
    modelSelect.replaceChildren();
    for (const model of models) modelSelect.add(new Option(model.name, model.name));
    if (!models.length) {
      modelSelect.add(new Option("No models installed", ""));
      setConnection(true, "Ollama connected · no models");
    } else {
      if (models.some(model => model.name === previous)) modelSelect.value = previous;
      setConnection(true, "Ollama connected");
    }
    updateSendState();
  } catch {
    modelSelect.replaceChildren(new Option("Ollama unavailable", ""));
    setConnection(false, "Ollama not connected");
    updateSendState();
  }
}
function updateSendState() { sendButton.disabled = busy || !modelSelect.value || !promptInput.value.trim(); const stop = $("#stop-generation"); if (stop) stop.hidden = !busy; }
async function sendMessage(text) {
  const model = modelSelect.value;
  if (!model || busy) return;
  if (!activeChat()) makeChat();
  const chat = activeChat();
  addMessage("user", text);
  // Capture context before appending the assistant placeholder.
  const requestMessages = await buildRequestMessages(chat);
  const assistantMessage = addMessage("assistant", "Thinking…");
  assistantMessage.pending = true;
  busy = true; updateSendState();
  abortController = new AbortController();
  try {
    const response = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: requestMessages,
        stream: true,
        options: { temperature: Number(state.settings.temperature) }
      }),
      signal: abortController.signal
    });
    if (!response.ok) throw new Error((await response.text()) || `Request failed (${response.status})`);
    assistantMessage.content = "";
    persist(); renderMessages();
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n"); buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const chunk = JSON.parse(line);
        if (chunk.message?.content) {
          assistantMessage.content += chunk.message.content;
          const last = messagesEl.lastElementChild?.querySelector(".message-content");
          if (last && activeChat()?.id === chat.id) { last.textContent = assistantMessage.content; messagesEl.scrollTop = messagesEl.scrollHeight; }
        }
      }
    }
    if (buffer.trim()) {
      const chunk = JSON.parse(buffer);
      if (chunk.message?.content) assistantMessage.content += chunk.message.content;
    }
    if (!assistantMessage.content) assistantMessage.content = "(No response)";
    persist();
    await updateLongTermMemory(text, assistantMessage.content, model);
  } catch (error) {
    assistantMessage.content = error.name === "AbortError" ? "Response stopped." : `Couldn't reach Ollama: ${error.message}. Check that Ollama is running and allows this page's origin.`;
    renderMessages(); persist();
  } finally {
    busy = false; abortController = null; renderAll(); updateSendState(); promptInput.focus();
  }
}
form.addEventListener("submit", async event => {
  event.preventDefault();
  const text = promptInput.value.trim();
  if (!text || busy || !modelSelect.value) return;
  promptInput.value = ""; promptInput.style.height = "auto";
  await sendMessage(text);
});
promptInput.addEventListener("input", () => {
  promptInput.style.height = "auto";
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 180)}px`;
  updateSendState();
});
promptInput.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
});
modelSelect.addEventListener("change", updateSendState);
$("#stop-generation").addEventListener("click", () => abortController?.abort());
$("#new-chat").addEventListener("click", () => { if (!busy) makeChat(); });
$("#settings-open").addEventListener("click", () => { renderAll(); $("#settings-overlay").hidden = false; });
$("#memory-open").addEventListener("click", () => { renderMemories(); $("#memory-overlay").hidden = false; });
$("#mobile-menu").addEventListener("click", () => $("#sidebar").classList.toggle("mobile-open"));
document.querySelectorAll("[data-close]").forEach(button => button.addEventListener("click", () => { $("#" + button.dataset.close).hidden = true; }));
document.querySelectorAll(".overlay").forEach(overlay => overlay.addEventListener("click", event => { if (event.target === overlay) overlay.hidden = true; }));
$("#settings-done").addEventListener("click", () => {
  state.settings.contextSize = Number($("#context-size").value);
  state.settings.memoryEnabled = $("#memory-enabled").checked;
  state.settings.autoMemory = $("#auto-memory").checked;
  state.settings.temperature = Number($("#temperature").value);
  state.settings.embeddingModel = $("#embedding-model").value.trim() || "nomic-embed-text";
  persist(); renderAll(); $("#settings-overlay").hidden = true;
});
$("#export-data").addEventListener("click", () => {
  const data = { format: "Ultillama backup", version: 1, exportedAt: new Date().toISOString(), ...state };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob); const link = document.createElement("a");
  link.href = url; link.download = "ultillama-backup.json"; link.click(); URL.revokeObjectURL(url);
});
$("#clear-chats").addEventListener("click", () => {
  if (!confirm("Delete every chat? Saved memories will be kept.")) return;
  state.chats = []; state.activeChatId = null; persist(); renderAll();
});
$("#memory-form").addEventListener("submit", event => {
  event.preventDefault();
  const input = $("#memory-input");
  const value = input.value.trim();
  if (!value) return;
  if (!state.memories.includes(value)) state.memories.push(value);
  delete state.memoryVectors[value];
  input.value = ""; persist(); renderMemories();
});
loadModels();
renderAll();
if (state.chats.length && !activeChat()) { state.activeChatId = state.chats[0].id; persist(); renderAll(); }
