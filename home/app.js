(() => {
  const $ = (selector) => document.querySelector(selector);
  const chatList = $("#chat-list");
  const messages = $("#messages");
  const welcome = $("#welcome-scene");
  const form = $("#chat-form");
  const input = $("#message-input");
  const count = $("#chat-count");
  const toast = $("#toast");
  const storageKey = "sol-conversations-v1";
  let chats = [];
  let activeId = null;
  let toastTimer;

  const icon = (name) => {
    const paths = {
      chat: '<path d="M20 11.5a7.5 7.5 0 0 1-7.5 7.5H5l1.8-3A7.5 7.5 0 1 1 20 11.5Z"/>',
      edit: '<path d="m14 5 5 5M4 20l4.2-.9L19 8.3 15.7 5 4.9 15.8 4 20Z"/>',
      trash: '<path d="M4 7h16M9 7V4h6v3m3 0-1 13H7L6 7m4 3v6m4-6v6"/>',
      user: '<circle cx="12" cy="8" r="3.5"/><path d="M5 20c.5-3.4 2.8-5 7-5s6.5 1.6 7 5"/>',
      sol: '<circle cx="12" cy="12" r="4"/><ellipse cx="12" cy="12" rx="10" ry="4.5" transform="rotate(-28 12 12)"/>'
    };
    return '<svg viewBox="0 0 24 24" aria-hidden="true">' + (paths[name] || paths.chat) + '</svg>';
  };
  const id = () => (crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2));
  const save = () => {
    try { localStorage.setItem(storageKey, JSON.stringify(chats)); }
    catch { showToast("Could not save conversations in this browser."); }
  };
  const showToast = (message) => {
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 2200);
  };
  const activeChat = () => chats.find(chat => chat.id === activeId);
  const createChat = () => {
    const chat = { id: id(), title: "New conversation", messages: [], created: Date.now() };
    chats.unshift(chat);
    activeId = chat.id;
    save();
    render();
    input.focus();
  };
  const renderList = () => {
    chatList.replaceChildren();
    count.textContent = String(chats.length).padStart(2, "0");
    chats.forEach(chat => {
      const item = document.createElement("div");
      item.className = "chat-item" + (chat.id === activeId ? " active" : "");
      item.tabIndex = 0;
      item.setAttribute("role", "button");
      item.setAttribute("aria-label", chat.title);
      const glyph = document.createElement("span");
      glyph.innerHTML = icon("chat");
      const title = document.createElement("span");
      title.className = "chat-title";
      title.textContent = chat.title;
      const actions = document.createElement("span");
      actions.className = "chat-actions";
      const rename = document.createElement("button");
      rename.className = "chat-action";
      rename.type = "button";
      rename.setAttribute("aria-label", "Rename conversation");
      rename.innerHTML = icon("edit");
      rename.addEventListener("click", event => {
        event.stopPropagation();
        const next = prompt("Rename conversation:", chat.title);
        if (next && next.trim()) { chat.title = next.trim().slice(0, 60); save(); renderList(); }
      });
      const remove = document.createElement("button");
      remove.className = "chat-action";
      remove.type = "button";
      remove.setAttribute("aria-label", "Delete conversation");
      remove.innerHTML = icon("trash");
      remove.addEventListener("click", event => {
        event.stopPropagation();
        if (!confirm("Delete this conversation?")) return;
        chats = chats.filter(entry => entry.id !== chat.id);
        if (activeId === chat.id) activeId = chats[0]?.id || null;
        save();
        render();
        if (!activeId) showToast("Conversation deleted.");
      });
      actions.append(rename, remove);
      item.append(glyph, title, actions);
      item.addEventListener("click", () => { activeId = chat.id; render(); });
      item.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activeId = chat.id; render(); }
      });
      chatList.append(item);
    });
  };
  const renderMessages = () => {
    const chat = activeChat();
    const hasMessages = !!chat && chat.messages.length > 0;
    welcome.hidden = hasMessages;
    messages.classList.toggle("visible", hasMessages);
    messages.replaceChildren();
    if (!hasMessages) return;
    chat.messages.forEach(message => {
      const row = document.createElement("article");
      row.className = "message " + message.role;
      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.innerHTML = icon(message.role === "user" ? "user" : "sol");
      const wrap = document.createElement("div");
      wrap.className = "bubble-wrap";
      const author = document.createElement("div");
      author.className = "message-author";
      author.textContent = message.role === "user" ? "YOU" : "SOL";
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = message.text;
      wrap.append(author, bubble);
      row.append(avatar, wrap);
      messages.append(row);
    });
    $("#conversation").scrollTop = $("#conversation").scrollHeight;
  };
  const render = () => { renderList(); renderMessages(); };
  const sendMessage = (text) => {
    const value = text.trim();
    if (!value) return;
    if (!activeChat()) createChat();
    const chat = activeChat();
    chat.messages.push({ role: "user", text: value });
    chat.messages.push({ role: "sol", text: "Sol has no brain yet. Come back later!" });
    if (chat.messages.filter(message => message.role === "user").length === 1) {
      chat.title = value.length > 30 ? value.slice(0, 27) + "..." : value;
    }
    save();
    render();
    input.value = "";
    input.style.height = "auto";
    input.focus();
  };
  form.addEventListener("submit", event => { event.preventDefault(); sendMessage(input.value); });
  input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 140) + "px"; });
  input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
  });
  $("#new-chat").addEventListener("click", createChat);
  $("#mobile-menu").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  document.addEventListener("keydown", event => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); createChat(); }
    if (event.key === "Escape") $("#sidebar").classList.remove("open");
  });
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "[]");
    if (Array.isArray(saved)) chats = saved.filter(chat => chat && typeof chat.id === "string" && Array.isArray(chat.messages));
  } catch { chats = []; }
  if (chats.length) activeId = chats[0].id;
  render();
})();