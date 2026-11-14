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
  const modelPicker = $("#model-picker");
  const modelMenu = $("#model-menu");
  const settingsKey = "astra-settings-v1";
  let settings = { animations: true, enterToSend: true };
  try {
    const storedSettings = JSON.parse(localStorage.getItem(settingsKey) || "{}");
    settings = { ...settings, ...storedSettings };
  } catch {}
  let reducedMotion = !settings.animations;
  document.documentElement.classList.toggle("motion-off", reducedMotion);
  const storageKey = "sol-conversations-v1";
  let chats = [];
  let activeId = null;
  let toastTimer;
  let animateNewConversation = false;

  const icon = (name) => '<img class="ui-icon" src="assets/icons/' + name + '.svg" alt="" aria-hidden="true">';
  const id = () => (crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2));
  const save = () => {
    try { localStorage.setItem(storageKey, JSON.stringify(chats)); }
    catch { showToast("Could not save conversations in this browser."); }
  };
  const saveSettings = () => {
    try { localStorage.setItem(settingsKey, JSON.stringify(settings)); }
    catch { showToast("Could not save settings in this browser."); }
  };
  const showToast = (message) => {
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 2200);
  };
  const activeChat = () => chats.find(chat => chat.id === activeId);
  const makeEmptyChat = () => ({ id: id(), title: "New conversation", messages: [], created: Date.now() });
  const ensureChatExists = () => {
    if (chats.length) return;
    const chat = makeEmptyChat();
    chats.unshift(chat);
    activeId = chat.id;
  };
  const selectChat = (nextId) => {
    if (nextId === activeId) return;
    const previous = activeChat();
    if (previous && previous.messages.length === 0) {
      chats = chats.filter(chat => chat.id !== previous.id);
      save();
    }
    activeId = nextId;
    render("switch");
    $("#sidebar").classList.remove("open");
  };
  const createChat = () => {
    // Keep only one blank conversation: discard any abandoned empty chats first.
    chats = chats.filter(existing => existing.messages.length > 0);
    const chat = makeEmptyChat();
    chats.unshift(chat);
    activeId = chat.id;
    animateNewConversation = true;
    save();
    render("new");
    $("#sidebar").classList.remove("open");
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
      item.addEventListener("click", () => selectChat(chat.id));
      item.addEventListener("keydown", event => {
        if (event.target !== item) return;
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectChat(chat.id);
        }
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
        ensureChatExists();
        save();
        render();
        closeModal();
        showToast("Conversation deleted.");
      })
    ]);
  };
  const makeSettingRow = (title, description, key) => {
    const row = document.createElement("div");
    row.className = "settings-row";
    const labels = document.createElement("div");
    const label = document.createElement("div");
    label.className = "settings-label";
    label.textContent = title;
    const note = document.createElement("div");
    note.className = "settings-note";
    note.textContent = description;
    labels.append(label, note);

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "toggle";
    toggle.setAttribute("role", "switch");
    toggle.setAttribute("aria-label", title);
    const sync = () => toggle.setAttribute("aria-checked", String(!!settings[key]));
    sync();
    toggle.addEventListener("click", () => {
      settings[key] = !settings[key];
      if (key === "animations") {
        reducedMotion = !settings.animations;
        document.documentElement.classList.toggle("motion-off", reducedMotion);
      }
      sync();
      saveSettings();
    });
    row.append(labels, toggle);
    return row;
  };

  const openClearAllDialog = () => {
    if (!chats.length) { showToast("There are no conversations to clear."); return; }
    const message = document.createElement("p");
    message.textContent = "Delete all " + chats.length + " conversations from this browser? This cannot be undone.";
    openModal("Clear all conversations?", message, [
      makeModalButton("Cancel", "", openSettingsDialog),
      makeModalButton("Delete all", "danger", () => {
        chats = [];
        activeId = null;
        ensureChatExists();
        save();
        closeModal();
        render("new");
        showToast("All conversations cleared.");
      })
    ]);
  };

  function openSettingsDialog() {
    const content = document.createElement("div");
    content.className = "settings-panel";
    content.append(
      makeSettingRow("Interface animations", "Enable motion throughout ASTRA", "animations"),
      makeSettingRow("Enter to send", "Use Enter to send; Shift + Enter for a new line", "enterToSend")
    );
    const clear = document.createElement("button");
    clear.type = "button";
    clear.className = "settings-danger-action";
    clear.textContent = "Clear all conversations";
    clear.addEventListener("click", openClearAllDialog);
    content.append(clear);
    openModal("Settings", content, [makeModalButton("Done", "primary", closeModal)]);
  };

  const buildMessage = (message, extraClass = "") => {
    const row = document.createElement("article");
    row.className = "message " + message.role + (extraClass ? " " + extraClass : "");
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.innerHTML = icon(message.role === "user" ? "user" : "astra");
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
  const renderMessages = (transition = "switch") => {
    const chat = activeChat();
    const hasMessages = !!chat && chat.messages.length > 0;
    welcome.hidden = hasMessages;
    messages.classList.toggle("visible", hasMessages);
    messages.replaceChildren();
    welcome.classList.remove("welcome-enter");
    messages.classList.remove("chat-switch-enter");
    if (!hasMessages) {
      if (transition === "new") {
        void welcome.offsetWidth;
        welcome.classList.add("welcome-enter");
      }
      return;
    }
    chat.messages.forEach(message => messages.append(buildMessage(message)));
    if (transition === "switch" && !reducedMotion) {
      void messages.offsetWidth;
      messages.classList.add("chat-switch-enter");
    }
    $("#conversation").scrollTop = $("#conversation").scrollHeight;
  };
  const render = (transition = "switch") => { renderList(); renderMessages(transition); };
  const easeScrollToBottom = (element, duration = 480) => new Promise(resolve => {
    const startY = element.scrollTop;
    const distance = element.scrollHeight - element.clientHeight - startY;
    if (Math.abs(distance) < 2 || reducedMotion) {
      element.scrollTop = element.scrollHeight;
      resolve();
      return;
    }
    const started = performance.now();
    const frame = now => {
      const progress = Math.min(1, (now - started) / duration);
      const eased = 1 - Math.pow(1 - progress, 4);
      element.scrollTop = startY + distance * eased;
      if (progress < 1) requestAnimationFrame(frame);
      else resolve();
    };
    requestAnimationFrame(frame);
  });

  const syncInputHeight = () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 7 * 21) + "px";
    const box = input.closest(".input-box");
    const overflowing = input.scrollHeight > input.clientHeight + 1;
    box?.classList.toggle("input-overflow", overflowing);
  };

  let astraModel = null;
  let astraModelPromise = null;

  const loadAstraModel = () => {
    if (astraModel) return Promise.resolve(astraModel);
    if (!astraModelPromise) {
      $("#selected-model-status").textContent = "Loading model...";
      modelPicker.title = "Loading ASTRA 0.1 into this browser";
      astraModelPromise = window.AstraBrowserModel.load().then(model => {
        astraModel = model;
        $("#selected-model-status").textContent = "Ready · In browser";
        modelPicker.title = "ASTRA 0.1 is loaded in this browser";
        return model;
      }).catch(error => {
        astraModelPromise = null;
        $("#selected-model-status").textContent = "Load failed";
        modelPicker.title = error.message || "Could not load ASTRA 0.1";
        throw error;
      });
    }
    return astraModelPromise;
  };

  const buildPromptIds = (history, currentMessage, tokenizer) => {
    const maxPromptTokens = 432;
    const prefix = tokenizer.encode("<user>\n");
    const suffix = tokenizer.encode("\n</user>\n<astra>\n");
    let currentBody = tokenizer.encode(currentMessage);
    const currentRoom = Math.max(1, maxPromptTokens - prefix.length - suffix.length);
    if (currentBody.length > currentRoom) currentBody = currentBody.slice(-currentRoom);
    let promptIds = prefix.concat(currentBody, suffix);

    const turns = [];
    for (let i = 0; i < history.length; i++) {
      if (history[i].role !== "user") continue;
      const assistant = history[i + 1]?.role === "sol" ? history[i + 1].text : "";
      turns.push({ user: history[i].text, assistant });
    }

    for (let i = turns.length - 1; i >= 0; i--) {
      const turn = turns[i];
      const turnIds = tokenizer.encode(
        "<user>\n" + turn.user + "\n</user>\n<astra>\n" +
        turn.assistant + "\n</astra>\n"
      );
      if (promptIds.length + turnIds.length > maxPromptTokens) continue;
      promptIds = turnIds.concat(promptIds);
    }
    return promptIds;
  };

  const sendMessage = async (text) => {
    const value = text.trim();
    if (!value) return;
    if (!activeChat()) createChat();
    const chat = activeChat();
    const history = chat.messages.map(message => ({ role: message.role, text: message.text }));
    const userMessage = { role: "user", text: value };
    const botMessage = { role: "sol", text: "Thinking..." };
    chat.messages.push(userMessage, botMessage);
    if (chat.messages.filter(message => message.role === "user").length === 1) {
      chat.title = value.length > 30 ? value.slice(0, 27) + "..." : value;
      renderList();
    }

    const inputBox = $(".input-box");
    const composerShell = $(".composer-shell");
    const sendButton = $(".send-button");
    const sourceRect = inputBox.getBoundingClientRect();
    welcome.hidden = true;
    messages.classList.add("visible");

    const userRow = buildMessage(userMessage);
    userRow.classList.add("message-morph-target");
    messages.append(userRow);
    const bubble = userRow.querySelector(".bubble");
    const avatar = userRow.querySelector(".avatar");
    const conversation = $("#conversation");

    // Keep the actual textarea and its text inside the moving input box.
    inputBox.classList.remove("input-overflow");
    input.style.height = "24px";
    syncInputHeight();
    save();

    await easeScrollToBottom(conversation, 250);
    if (activeId !== chat.id) return;

    const targetRect = bubble.getBoundingClientRect();
    // Hold the original composer slot so the send button never shifts.
    const inputPlaceholder = document.createElement("div");
    inputPlaceholder.className = "input-box-placeholder";
    inputPlaceholder.style.width = sourceRect.width + "px";
    inputPlaceholder.style.height = sourceRect.height + "px";
    inputPlaceholder.style.flex = "0 0 " + sourceRect.width + "px";
    composerShell.insertBefore(inputPlaceholder, inputBox);
    const restoreComposer = () => {
      if (inputBox.parentElement !== composerShell) composerShell.insertBefore(inputBox, sendButton);
      inputPlaceholder.remove();
      inputBox.classList.remove("input-box-flight", "input-overflow");
      inputBox.removeAttribute("style");
      input.value = "";
      input.style.height = "24px";
      syncInputHeight();
    };

    // Lift the real input box into a fixed layer without replacing its contents.
    Object.assign(inputBox.style, {
      position: "fixed",
      zIndex: "50",
      left: sourceRect.left + "px",
      top: sourceRect.top + "px",
      width: sourceRect.width + "px",
      height: sourceRect.height + "px",
      margin: "0",
      flex: "none",
      pointerEvents: "none",
      transform: "none"
    });
    inputBox.classList.add("input-box-flight");
    inputBox.style.overflow = "hidden";
    document.body.append(inputBox);

    let activeFlightAnimation = null;
    let inputHeightAnimation = null;
    try {
      // Move and contract the real composer together: one continuous morph.
      userRow.style.visibility = "visible";
      userRow.classList.add("morph-avatar-in");
      const sourceStyle = getComputedStyle(inputBox);
      const targetStyle = getComputedStyle(bubble);
      const inputStyle = getComputedStyle(input);
      const targetInputHeight = Math.max(1,
        targetRect.height -
        parseFloat(targetStyle.paddingTop) -
        parseFloat(targetStyle.paddingBottom) -
        parseFloat(targetStyle.borderTopWidth) -
        parseFloat(targetStyle.borderBottomWidth)
      );
      inputBox.style.backgroundImage = "linear-gradient(135deg, rgba(54,82,119,0), rgba(40,57,91,0))";
      const duration = reducedMotion ? 1 : 350;
      inputHeightAnimation = input.animate([
        {
          height: input.getBoundingClientRect().height + "px",
          minHeight: inputStyle.minHeight,
          paddingTop: inputStyle.paddingTop,
          paddingBottom: inputStyle.paddingBottom
        },
        {
          height: targetInputHeight + "px",
          minHeight: targetInputHeight + "px",
          paddingTop: "0px",
          paddingBottom: "0px"
        }
      ], {
        duration,
        easing: "cubic-bezier(.22,.78,.18,1)",
        fill: "forwards"
      });
      activeFlightAnimation = inputBox.animate([
        {
          left: sourceRect.left + "px",
          top: sourceRect.top + "px",
          width: sourceRect.width + "px",
          height: sourceRect.height + "px",
          minHeight: sourceStyle.minHeight,
          backgroundColor: sourceStyle.backgroundColor,
          backgroundImage: "linear-gradient(135deg, rgba(54,82,119,0), rgba(40,57,91,0))",
          borderColor: sourceStyle.borderColor,
          borderRadius: sourceStyle.borderRadius,
          boxShadow: sourceStyle.boxShadow,
          padding: sourceStyle.padding
        },
        {
          left: targetRect.left + "px",
          top: targetRect.top + "px",
          width: targetRect.width + "px",
          height: targetRect.height + "px",
          minHeight: targetRect.height + "px",
          backgroundColor: targetStyle.backgroundColor,
          backgroundImage: targetStyle.backgroundImage,
          borderColor: targetStyle.borderColor,
          borderRadius: targetStyle.borderRadius,
          boxShadow: targetStyle.boxShadow,
          padding: targetStyle.padding
        }
      ], {
        duration,
        easing: "cubic-bezier(.22,.78,.18,1)",
        fill: "forwards"
      });
      await Promise.all([activeFlightAnimation.finished, inputHeightAnimation.finished]);
      activeFlightAnimation.cancel();
      inputHeightAnimation.cancel();
      activeFlightAnimation = null;
      inputHeightAnimation = null;
      if (activeId !== chat.id) throw new Error("Conversation changed");

      // Swap directly to the message; only the restored, empty composer fades in.
      restoreComposer();
      userRow.classList.add("morph-message-in");
      inputBox.classList.add("composer-reform");
      window.setTimeout(() => inputBox.classList.remove("composer-reform"), reducedMotion ? 1 : 440);

      window.setTimeout(() => {
        if (activeId !== chat.id) return;
        const botRow = buildMessage(botMessage, "bot-arriving");
        messages.append(botRow);
        requestAnimationFrame(() => botRow.classList.add("bot-visible"));
        easeScrollToBottom(conversation, 360);
        (async () => {
          try {
            const model = await loadAstraModel();
            const promptIds = buildPromptIds(history, value, model.tokenizer);
            const reply = await model.generate(promptIds, { maxNewTokens: 80 });
            if (!reply) throw new Error("ASTRA did not generate a response. Try sending that again.");
            botMessage.text = reply;
            botRow.querySelector(".bubble").textContent = reply;
            save();
            easeScrollToBottom(conversation, 260);
          } catch (error) {
            botMessage.text = error.message || "ASTRA could not generate a response.";
            botRow.querySelector(".bubble").textContent = botMessage.text;
            save();
            easeScrollToBottom(conversation, 260);
          }
        })();
      }, 100);
    } catch {
      activeFlightAnimation?.cancel();
      inputHeightAnimation?.cancel();
      activeFlightAnimation = null;
      inputHeightAnimation = null;
      restoreComposer();
      if (activeId === chat.id) {
        userRow.style.visibility = "visible";
        userRow.classList.add("morph-message-in");
      }
    }
    input.focus();
  };
  form.addEventListener("submit", event => { event.preventDefault(); sendMessage(input.value); });
  document.querySelectorAll(".idea-card[data-prompt]").forEach(card => {
    card.addEventListener("click", () => {
      input.value = card.dataset.prompt || "";
      syncInputHeight();
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
    });
  });
  input.addEventListener("input", syncInputHeight);
  input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey && settings.enterToSend) {
      event.preventDefault();
      form.requestSubmit();
    }
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
  modelPicker.addEventListener("click", () => {
    const opening = modelMenu.hidden;
    modelMenu.hidden = !opening;
    modelPicker.setAttribute("aria-expanded", String(opening));
  });
  modelMenu.addEventListener("click", event => {
    const option = event.target.closest("[data-model]");
    if (!option) return;
    modelMenu.querySelectorAll("[data-model]").forEach(item => {
      const selected = item === option;
      item.classList.toggle("selected", selected);
      item.setAttribute("aria-selected", String(selected));
    });
    $("#selected-model-name").textContent = option.querySelector("strong").textContent;
    modelMenu.hidden = true;
    modelPicker.setAttribute("aria-expanded", "false");
  });
  $("#settings-button").addEventListener("click", openSettingsDialog);
  $("#modal-close").addEventListener("click", closeModal);
  modalBackdrop.addEventListener("click", event => { if (event.target === modalBackdrop) closeModal(); });
  document.addEventListener("click", event => {
    if (!event.target.closest("#model-picker-wrap")) {
      modelMenu.hidden = true;
      modelPicker.setAttribute("aria-expanded", "false");
    }
    if (!event.target.closest(".edit-menu-wrap")) {
      editMenu.hidden = true;
      editButton.setAttribute("aria-expanded", "false");
    }
  });
  $("#mobile-menu").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  document.addEventListener("keydown", event => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") { event.preventDefault(); createChat(); }
    if (event.key === "Tab" && !modalBackdrop.hidden) {
      const focusable = [...modalBackdrop.querySelectorAll("button:not([disabled]), input:not([disabled]), textarea:not([disabled])")];
      if (focusable.length) {
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }
    }
    if (event.key === "Escape") {
      $("#sidebar").classList.remove("open");
      editMenu.hidden = true;
      editButton.setAttribute("aria-expanded", "false");
      modelMenu.hidden = true;
      modelPicker.setAttribute("aria-expanded", "false");
      if (!modalBackdrop.hidden) closeModal();
    }
  });
  loadAstraModel().catch(() => {});
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "[]");
    if (Array.isArray(saved)) chats = saved.filter(chat => chat && typeof chat.id === "string" && Array.isArray(chat.messages));
  } catch { chats = []; }
  ensureChatExists();
  if (chats.length && !activeId) activeId = chats[0].id;
  save();
  render("initial");
})();