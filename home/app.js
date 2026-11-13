(() => {
  const $ = (selector) => document.querySelector(selector);
  const chatList = $("#chat-list");
  const messages = $("#messages");
  const welcome = $("#welcome-scene");
  const form = $("#chat-form");
  const input = $("#message-input");
  const count = $("#chat-count");
  const toast = $("#toast");
  const modalBackdrop = $("#modal-backdrop");
  const modalTitle = $("#modal-title");
  const modalContent = $("#modal-content");
  const modalActions = $("#modal-actions");
  const editButton = $("#edit-chat-button");
  const editMenu = $("#edit-chat-menu");
  let reducedMotion = false;
  const storageKey = "sol-conversations-v1";
  let chats = [];
  let activeId = null;
  let toastTimer;
  let animateNewConversation = false;

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
    animateNewConversation = true;
    save();
    render();
    input.focus();
  };
  const renderList = () => {
    chatList.replaceChildren();
    count.textContent = String(chats.length).padStart(2, "0");
    chats.forEach(chat => {
      const item = document.createElement("div");
      item.className = "chat-item" + (chat.id === activeId ? " active" : "") + (animateNewConversation ? (chat.id === activeId ? " entering" : " shifting") : "");
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
        activeId = chat.id;
        openRenameDialog();
      });
      const remove = document.createElement("button");
      remove.className = "chat-action";
      remove.type = "button";
      remove.setAttribute("aria-label", "Delete conversation");
      remove.innerHTML = icon("trash");
      remove.addEventListener("click", event => {
        event.stopPropagation();
        activeId = chat.id;
        openDeleteDialog();
      });
      actions.append(rename, remove);
      item.append(glyph, title, actions);
      item.addEventListener("click", () => { activeId = chat.id; render(); });
      item.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activeId = chat.id; render(); }
      });
      chatList.append(item);
    });
    animateNewConversation = false;
  };

  const closeModal = () => {
    modalBackdrop.hidden = true;
    modalContent.replaceChildren();
    modalActions.replaceChildren();
  };
  const openModal = (title, contentNode, actions) => {
    modalTitle.textContent = title;
    modalContent.replaceChildren(contentNode);
    modalActions.replaceChildren(...actions);
    modalBackdrop.hidden = false;
  };
  const makeModalButton = (label, kind, action) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "modal-button" + (kind ? " " + kind : "");
    button.textContent = label;
    button.addEventListener("click", action);
    return button;
  };
  const openRenameDialog = () => {
    const chat = activeChat();
    if (!chat) { showToast("Choose a conversation first."); return; }
    const inputField = document.createElement("input");
    inputField.className = "modal-input";
    inputField.maxLength = 60;
    inputField.value = chat.title;
    inputField.setAttribute("aria-label", "Conversation name");
    const saveName = () => {
      const next = inputField.value.trim();
      if (!next) { inputField.focus(); return; }
      chat.title = next;
      save();
      renderList();
      closeModal();
      showToast("Conversation renamed.");
    };
    openModal("Rename conversation", inputField, [
      makeModalButton("Cancel", "", closeModal),
      makeModalButton("Save changes", "primary", saveName)
    ]);
    inputField.addEventListener("keydown", event => { if (event.key === "Enter") saveName(); });
    inputField.focus();
    inputField.select();
  };
  const openDeleteDialog = () => {
    const chat = activeChat();
    if (!chat) { showToast("Choose a conversation first."); return; }
    const message = document.createElement("p");
    message.textContent = "Delete “" + chat.title + "”? This conversation will be removed from this browser.";
    openModal("Delete conversation?", message, [
      makeModalButton("Keep chat", "", closeModal),
      makeModalButton("Delete", "danger", () => {
        chats = chats.filter(entry => entry.id !== chat.id);
        if (activeId === chat.id) activeId = chats[0]?.id || null;
        save();
        render();
        closeModal();
        showToast("Conversation deleted.");
      })
    ]);
  };
  const openSettingsDialog = () => {
    const content = document.createElement("div");
    const row = document.createElement("div");
    row.className = "settings-row";
    const labels = document.createElement("div");
    const label = document.createElement("div");
    label.className = "settings-label";
    label.textContent = "Motion";
    const note = document.createElement("div");
    note.className = "settings-note";
    note.textContent = "Enable interface animations";
    labels.append(label, note);
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "toggle";
    toggle.setAttribute("role", "switch");
    toggle.setAttribute("aria-label", "Enable animations");
    toggle.setAttribute("aria-checked", String(!reducedMotion));
    const applyMotion = () => {
      reducedMotion = !reducedMotion;
      toggle.setAttribute("aria-checked", String(!reducedMotion));
      document.documentElement.classList.toggle("motion-off", reducedMotion);
    };
    toggle.addEventListener("click", applyMotion);
    row.append(labels, toggle);
    content.append(row);
    openModal("Settings", content, [makeModalButton("Done", "primary", closeModal)]);
  };

  const buildMessage = (message, extraClass = "") => {
    const row = document.createElement("article");
    row.className = "message " + message.role + (extraClass ? " " + extraClass : "");
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.innerHTML = icon(message.role === "user" ? "user" : "sol");
    const wrap = document.createElement("div");
    wrap.className = "bubble-wrap";
    const author = document.createElement("div");
    author.className = "message-author";
    author.textContent = message.role === "user" ? "YOU" : "ASTRA";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = message.text;
    wrap.append(author, bubble);
    row.append(avatar, wrap);
    return row;
  };
  const renderMessages = () => {
    const chat = activeChat();
    const hasMessages = !!chat && chat.messages.length > 0;
    welcome.hidden = hasMessages;
    messages.classList.toggle("visible", hasMessages);
    messages.replaceChildren();
    if (!hasMessages) return;
    chat.messages.forEach(message => messages.append(buildMessage(message)));
    $("#conversation").scrollTop = $("#conversation").scrollHeight;
  };
  const render = () => { renderList(); renderMessages(); };
  const sendMessage = (text) => {
    const value = text.trim();
    if (!value) return;
    if (!activeChat()) createChat();
    const chat = activeChat();
    const userMessage = { role: "user", text: value };
    const botMessage = { role: "sol", text: "Astra has no brain yet. Come back later!" };
    chat.messages.push(userMessage, botMessage);
    if (chat.messages.filter(message => message.role === "user").length === 1) {
      chat.title = value.length > 30 ? value.slice(0, 27) + "..." : value;
      renderList();
    }

    const sourceRect = input.getBoundingClientRect();
    const wasWelcome = !messages.classList.contains("visible");
    welcome.hidden = true;
    messages.classList.add("visible");
    const userRow = buildMessage(userMessage, "message-arriving");
    messages.append(userRow);
    const targetRect = userRow.getBoundingClientRect();
    const dx = sourceRect.left - targetRect.left;
    const dy = sourceRect.top - targetRect.top;
    userRow.style.setProperty("--flight-x", dx + "px");
    userRow.style.setProperty("--flight-y", dy + "px");
    userRow.classList.add("message-flight");
    if (wasWelcome) messages.classList.add("first-message");

    input.value = "";
    input.style.height = "24px";
    form.classList.remove("input-refresh");
    void form.offsetWidth;
    form.classList.add("input-refresh");
    requestAnimationFrame(() => {
      userRow.classList.add("flight-start");
      userRow.classList.remove("message-arriving");
      $("#conversation").scrollTop = $("#conversation").scrollHeight;
    });

    save();
    window.setTimeout(() => {
      if (activeId !== chat.id) return;
      userRow.classList.remove("message-flight", "flight-start");
      userRow.style.removeProperty("--flight-x");
      userRow.style.removeProperty("--flight-y");
      const botRow = buildMessage(botMessage, "bot-arriving");
      messages.append(botRow);
      requestAnimationFrame(() => botRow.classList.add("bot-visible"));
      $("#conversation").scrollTop = $("#conversation").scrollHeight;
    }, 520);
    input.focus();
  };
  form.addEventListener("submit", event => { event.preventDefault(); sendMessage(input.value); });
  input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 140) + "px"; });
  input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); form.requestSubmit(); }
  });
  $("#new-chat").addEventListener("click", createChat);
  editButton.addEventListener("click", () => {
    const opening = editMenu.hidden;
    editMenu.hidden = !opening;
    editButton.setAttribute("aria-expanded", String(opening));
  });
  editMenu.addEventListener("click", event => {
    const action = event.target.closest("button")?.dataset.action;
    if (!action) return;
    editMenu.hidden = true;
    editButton.setAttribute("aria-expanded", "false");
    if (!activeChat()) { showToast("Choose a conversation first."); return; }
    if (action === "rename") openRenameDialog();
    if (action === "delete") openDeleteDialog();
  });
  $("#settings-button").addEventListener("click", openSettingsDialog);
  $("#modal-close").addEventListener("click", closeModal);
  modalBackdrop.addEventListener("click", event => { if (event.target === modalBackdrop) closeModal(); });
  document.addEventListener("click", event => {
    if (!event.target.closest(".edit-menu-wrap")) {
      editMenu.hidden = true;
      editButton.setAttribute("aria-expanded", "false");
    }
  });
  $("#mobile-menu").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  document.addEventListener("keydown", event => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); createChat(); }
    if (event.key === "Escape") {
      $("#sidebar").classList.remove("open");
      editMenu.hidden = true;
      editButton.setAttribute("aria-expanded", "false");
      if (!modalBackdrop.hidden) closeModal();
    }
  });
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "[]");
    if (Array.isArray(saved)) chats = saved.filter(chat => chat && typeof chat.id === "string" && Array.isArray(chat.messages));
  } catch { chats = []; }
  if (chats.length) activeId = chats[0].id;
  render();
})();