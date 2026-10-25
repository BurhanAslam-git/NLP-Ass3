(function () {
  const WS_PATH = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/chat`;
  const STORAGE_CONVERSATIONS = "opac_fast_nuces_conversations_v1";
  const STORAGE_LEGACY_SESSION = "opac_fast_nuces_session_id";
  const STORAGE_LEGACY_LOG = "opac_fast_nuces_chat_log";
  const DEFAULT_CHAT_TITLE = "New chat";
  const TITLE_MAX_LEN = 52;
  const MAX_CHARS = 4000;
  const REPLY_TIMEOUT_MS = 125000;
  const HEALTH_POLL_MS = 30000;
  // Safety net only: the reply normally ends on the server's "done" frame.
  // (Was 600 ms, which cut replies off whenever the CPU model paused or ran a tool.)
  const STREAM_IDLE_MS = 20000;
  const MAX_STORED_CONVERSATIONS = 48;

  const messageList = document.getElementById("message-list");
  const form = document.getElementById("chat-form");
  const input = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const newSessionBtn = document.getElementById("new-session-btn");
  const newChatSidebarBtn = document.getElementById("new-chat-sidebar-btn");
  const chatListEl = document.getElementById("chat-list");
  const chatHeading = document.getElementById("chat-heading");
  const retryBtn = document.getElementById("retry-btn");
  const statusEl = document.getElementById("connection-status");
  const connectionLine = document.getElementById("connection-line");
  const thinkingEl = document.getElementById("thinking-indicator");
  const thinkingLabel = document.getElementById("thinking-label");
  const starterPrompts = document.getElementById("starter-prompts");
  const modelStatus = document.getElementById("model-status");
  const systemBanner = document.getElementById("system-banner");
  const charCount = document.getElementById("char-count");
  const composerDisclaimer = document.getElementById("composer-disclaimer");
  const startersDisclosure = document.querySelector(".sidebar-disclosure");
  const appModal = document.getElementById("app-modal");
  const appModalTitle = document.getElementById("app-modal-title");
  const appModalMessage = document.getElementById("app-modal-message");
  const appModalInputWrap = document.getElementById("app-modal-input-wrap");
  const appModalInput = document.getElementById("app-modal-input");
  const appModalCancel = document.getElementById("app-modal-cancel");
  const appModalConfirm = document.getElementById("app-modal-confirm");

  // Ass 2 elements
  const contextBarStats = document.getElementById("context-bar-stats");
  const contextBarFill = document.getElementById("context-bar-fill");
  const btnSummarise = document.getElementById("btn-summarise");
  const toastContainer = document.getElementById("toast-container");
  const btnViewSources = document.getElementById("btn-view-sources");
  const sourcesModal = document.getElementById("sources-modal");
  const btnCloseSourcesModal = document.getElementById("btn-close-sources-modal");
  const sourcesModalBackdrop = document.getElementById("sources-modal-backdrop");
  const sourcesList = document.getElementById("sources-list");
  const sourceTitleDisplay = document.getElementById("source-title-display");
  const sourceContentDisplay = document.getElementById("source-content-display");

  // Ass 3 voice elements
  const micBtn = document.getElementById("mic-btn");
  const partialTranscript = document.getElementById("partial-transcript");
  const speakToggle = document.getElementById("speak-toggle");
  const stopAudioBtn = document.getElementById("stop-audio-btn");

  const STARTERS = [
    "Is Speech and Language Processing by Jurafsky available, and where is it located?",
    "What are the overdue fines for regular circulating books versus reserved textbooks?",
    "My card number is P-78421. What books do I currently have checked out and when are they due?",
    "Card P-78421 here. Please renew The Great Gatsby for me.",
    "Card P-78421. I want to borrow Clean Code (BK-1004).",
    "Where is the Quiet Study Zone located and what are the quiet hours?",
    "Do you have a copy of Harry Potter and the Sorcerer's Stone in the library?",
    "Can you write a Python script to sort an array using quicksort?",
  ];

  let sessionId = crypto.randomUUID();
  let chatLog = [];
  let conversationStore = { activeId: "", conversations: [] };
  let socket = null;
  let streaming = false;
  let reconnectTimer = null;
  let healthTimer = null;
  let replyTimer = null;
  let streamIdleTimer = null;
  let streamingAssistantBody = null;
  let streamingMessageEl = null;
  let libraryInfo = null;
  let activeRequestId = 0;
  let inflightSessionId = null;
  let modelHealthy = false;
  let manualDisconnect = false;
  let openChatMenuEl = null;
  let modalResolver = null;

  // Voice: mic capture + reply playback (see voice.js)
  const voice = window.createVoice({
    getSocket: function () {
      return socket;
    },
    getSessionId: function () {
      return sessionId;
    },
    onRecordingChange: function (recording) {
      micBtn.classList.toggle("recording", recording);
      micBtn.setAttribute("aria-pressed", recording ? "true" : "false");
      micBtn.textContent = recording ? "⏺ Stop" : "🎤 Speak";
      input.disabled = recording || streaming;
      if (recording) {
        showPartial("Listening… speak now. I'll stop when you pause.");
      }
    },
    onError: function (msg) {
      showPartial("");
      appendMessage("error", msg, { extraClass: "message-error" });
    },
  });

  function defaultGreeting() {
    return libraryInfo && libraryInfo.greeting
      ? libraryInfo.greeting
      : "Welcome. I can help with catalogue search, borrowing rules, due dates, and renewals.";
  }

  function setBanner(text, level) {
    if (!text) {
      systemBanner.hidden = true;
      systemBanner.textContent = "";
      systemBanner.className = "system-banner";
      return;
    }
    systemBanner.hidden = false;
    systemBanner.textContent = text;
    systemBanner.className = "system-banner system-banner-" + (level || "info");
  }

  function showToast(msg) {
    if (!toastContainer) return;
    const t = document.createElement("div");
    t.className = "toast";
    t.textContent = msg;
    toastContainer.appendChild(t);
    setTimeout(() => {
      t.classList.add("fade-out");
      setTimeout(() => t.remove(), 500);
    }, 3000);
  }

  function setStatus(state) {
    statusEl.classList.remove("status-online", "status-offline", "status-connecting");
    if (state === "open") {
      statusEl.textContent = "Connected";
      statusEl.classList.add("status-online");
      connectionLine.textContent = "Secure session with the virtual reference desk";
      retryBtn.hidden = true;
      setBanner("", "");
    } else if (state === "connecting") {
      statusEl.textContent = "Connecting";
      statusEl.classList.add("status-connecting");
      connectionLine.textContent = "Opening secure chat channel...";
      retryBtn.hidden = true;
    } else {
      statusEl.textContent = "Offline";
      statusEl.classList.add("status-offline");
      connectionLine.textContent = "Connection lost. Messages are paused until reconnect.";
      retryBtn.hidden = false;
      if (streaming) {
        handleStreamInterrupted("Connection lost before the reply finished.");
      }
      setBanner("Chat is offline. Click Reconnect or wait for automatic retry.", "warn");
    }
  }

  function assistantThinkingLabel() {
    return (libraryInfo ? libraryInfo.assistant_name : "NUCES Virtual Desk") + " is thinking";
  }

  function setThinking(visible) {
    thinkingLabel.textContent = assistantThinkingLabel();
    thinkingEl.hidden = !visible;
  }

  // Latency panel: one row per pipeline stage of the last turn (ms).
  const METRIC_LABELS = {
    asr_ms: "ASR (end of speech → final transcript)",
    retrieval_ms: "Retrieval (RAG)",
    tool_ms: "Tools",
    llm_first_token_ms: "LLM first token",
    tts_first_audio_ms: "TTS first sentence",
    eos_to_first_audio_ms: "End of speech → first audio",
  };

  function showMetrics(data) {
    const list = document.getElementById("metrics-list");
    list.innerHTML = "";
    Object.keys(METRIC_LABELS).forEach(function (key) {
      const dt = document.createElement("dt");
      dt.textContent = METRIC_LABELS[key];
      const dd = document.createElement("dd");
      dd.textContent = data[key] === null || data[key] === undefined ? "—" : Math.round(data[key]) + " ms";
      list.appendChild(dt);
      list.appendChild(dd);
    });
  }

  function showPartial(text) {
    partialTranscript.hidden = !text;
    partialTranscript.textContent = text || "";
  }

  function setStreamingControls(active) {
    sendBtn.disabled = active || !canSend();
    input.disabled = active;
    micBtn.disabled = active;
    if (btnSummarise) {
      btnSummarise.disabled = active;
    }
    updateStarterDisabled();
  }

  function canSend() {
    const text = input.value.trim();
    return text.length > 0 && text.length <= MAX_CHARS && socket && socket.readyState === WebSocket.OPEN;
  }

  function updateCharCount() {
    const len = input.value.length;
    charCount.textContent = len + " / " + MAX_CHARS;
    charCount.classList.toggle("char-count-limit", len >= MAX_CHARS - 50);
    sendBtn.disabled = streaming || !canSend();
  }

  function formatTime(date) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function formatChatLastActive(iso) {
    if (!iso) {
      return "";
    }
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) {
      return "";
    }
    const now = new Date();
    if (d.toDateString() === now.toDateString()) {
      return formatTime(d);
    }
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    if (d.toDateString() === yesterday.toDateString()) {
      return "Yesterday";
    }
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  }

  function getActiveConversation() {
    return conversationStore.conversations.find(function (c) {
      return c.id === conversationStore.activeId;
    });
  }

  function truncateTitle(text) {
    const t = text.trim();
    if (t.length <= TITLE_MAX_LEN) {
      return t;
    }
    return t.slice(0, TITLE_MAX_LEN - 1).trim() + "…";
  }

  function truncateWords(text, maxWords) {
    const cleaned = text.replace(/[?.!]+$/g, "").trim();
    const words = cleaned.split(/\s+/).slice(0, maxWords).join(" ");
    return truncateTitle(words);
  }

  function deriveChatTitle(text) {
    const t = text.replace(/\s+/g, " ").trim();
    if (!t) {
      return DEFAULT_CHAT_TITLE;
    }
    if (/who are you|what are you|your name/i.test(t)) {
      return "Desk identity";
    }
    if (/card\s+p[-\s]?\d+|p-\d+/i.test(t)) {
      const card = t.match(/p-\d+/i);
      if (/due|loan|renew/i.test(t)) {
        return card ? "Account " + card[0].toUpperCase() : "Patron account";
      }
      return card ? "Card " + card[0].toUpperCase() : "Patron lookup";
    }
    if (/renew/i.test(t)) {
      const book = t.match(/renew\s+(?:the\s+)?(.+?)(?:\?|on\s+card|$)/i);
      return book ? truncateTitle("Renewal · " + truncateWords(book[1], 3)) : "Renewal request";
    }
    if (/fine|overdue/i.test(t)) {
      return "Overdue fines";
    }
    if (/loan period|how long can i borrow/i.test(t)) {
      return "Loan period";
    }
    if (/hold/i.test(t)) {
      return "Hold policy";
    }
    if (/borrow|policy|circulation/i.test(t)) {
      return "Borrowing policy";
    }
    if (/due date|due this week|what is due/i.test(t)) {
      return "Due dates";
    }
    if (/do you have|in the catalogue|available/i.test(t)) {
      const titleMatch = t.match(/do you have\s+(.+?)\?*$/i);
      if (titleMatch) {
        return truncateTitle("Catalogue · " + truncateWords(titleMatch[1], 4));
      }
      return "Catalogue search";
    }
    if (/my name|who am i/i.test(t)) {
      return "Patron identity";
    }
    if (/joke|weather|homework|code/i.test(t)) {
      return "Off-topic check";
    }

    return truncateWords(t, 4);
  }

  function isConversationEmpty(conv) {
    if (!conv || !Array.isArray(conv.messages)) {
      return true;
    }
    return !conv.messages.some(function (m) {
      return m.role === "user";
    });
  }

  function findEmptyChat() {
    return conversationStore.conversations.find(function (c) {
      return isConversationEmpty(c);
    });
  }

  function closeAllChatMenus() {
    if (openChatMenuEl) {
      openChatMenuEl.hidden = true;
      const btn = openChatMenuEl.previousElementSibling;
      if (btn && btn.setAttribute) {
        btn.setAttribute("aria-expanded", "false");
      }
      openChatMenuEl = null;
    }
  }

  function closeModal(result) {
    if (!modalResolver) {
      return;
    }
    appModal.hidden = true;
    const resolve = modalResolver;
    modalResolver = null;
    resolve(result);
  }

  function showConfirm(options) {
    options = options || {};
    return new Promise(function (resolve) {
      modalResolver = resolve;
      appModalTitle.textContent = options.title || "Confirm";
      appModalMessage.textContent = options.message || "";
      appModalInputWrap.hidden = true;
      appModalConfirm.textContent = options.confirmLabel || "OK";
      appModalCancel.textContent = options.cancelLabel || "Cancel";
      appModal.hidden = false;
      appModalConfirm.focus();
    });
  }

  function showPrompt(options) {
    options = options || {};
    return new Promise(function (resolve) {
      modalResolver = resolve;
      appModalTitle.textContent = options.title || "Rename chat";
      appModalMessage.textContent = options.message || "";
      appModalInputWrap.hidden = false;
      appModalInput.value = options.defaultValue || "";
      appModalConfirm.textContent = options.confirmLabel || "Save";
      appModalCancel.textContent = options.cancelLabel || "Cancel";
      appModal.hidden = false;
      appModalInput.focus();
      appModalInput.select();
    });
  }

  function bindModalControls() {
    appModalCancel.addEventListener("click", function () {
      closeModal(null);
    });
    appModalConfirm.addEventListener("click", function () {
      if (!appModalInputWrap.hidden) {
        closeModal(appModalInput.value);
        return;
      }
      closeModal(true);
    });
    appModal.querySelectorAll("[data-modal-dismiss]").forEach(function (el) {
      el.addEventListener("click", function () {
        closeModal(appModalInputWrap.hidden ? false : null);
      });
    });
    appModalInput.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        closeModal(appModalInput.value);
      }
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && modalResolver) {
        closeModal(appModalInputWrap.hidden ? false : null);
      }
    });
  }

  document.addEventListener("click", function (e) {
    if (!openChatMenuEl) {
      return;
    }
    if (e.target.closest(".chat-list-menu-wrap")) {
      return;
    }
    closeAllChatMenus();
  });

  function resetServerSession(sessionIdToReset) {
    if (socket && socket.readyState === WebSocket.OPEN && sessionIdToReset) {
      socket.send(
        JSON.stringify({
          type: "reset_session",
          session_id: sessionIdToReset,
        })
      );
    }
  }

  function pruneConversationStore() {
    while (conversationStore.conversations.length > MAX_STORED_CONVERSATIONS) {
      const candidates = conversationStore.conversations.filter(function (c) {
        return c.id !== conversationStore.activeId;
      });
      if (candidates.length === 0) {
        break;
      }
      candidates.sort(function (a, b) {
        return new Date(a.updatedAt).getTime() - new Date(b.updatedAt).getTime();
      });
      const drop = candidates[0];
      resetServerSession(drop.id);
      conversationStore.conversations = conversationStore.conversations.filter(function (c) {
        return c.id !== drop.id;
      });
    }
  }

  function saveConversationStore() {
    try {
      pruneConversationStore();
      localStorage.setItem(STORAGE_CONVERSATIONS, JSON.stringify(conversationStore));
    } catch {
      /* private mode / quota */
    }
  }

  function loadConversationStore() {
    try {
      const raw = localStorage.getItem(STORAGE_CONVERSATIONS);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && Array.isArray(parsed.conversations)) {
          conversationStore = {
            activeId: parsed.activeId || "",
            conversations: parsed.conversations,
          };
          return true;
        }
      }
    } catch {
      /* ignore */
    }
    return false;
  }

  function migrateLegacySessionStorage() {
    try {
      const storedId = sessionStorage.getItem(STORAGE_LEGACY_SESSION);
      const raw = sessionStorage.getItem(STORAGE_LEGACY_LOG);
      if (!storedId || !raw) {
        return false;
      }
      const messages = JSON.parse(raw);
      if (!Array.isArray(messages) || messages.length === 0) {
        return false;
      }
      const now = new Date().toISOString();
      const firstUser = messages.find(function (m) {
        return m.role === "user";
      });
      conversationStore = {
        activeId: storedId,
        conversations: [
          {
            id: storedId,
            title: firstUser ? deriveChatTitle(firstUser.text) : DEFAULT_CHAT_TITLE,
            titleManual: false,
            createdAt: now,
            updatedAt: now,
            messages: messages,
          },
        ],
      };
      saveConversationStore();
      sessionStorage.removeItem(STORAGE_LEGACY_SESSION);
      sessionStorage.removeItem(STORAGE_LEGACY_LOG);
      return true;
    } catch {
      return false;
    }
  }

  function syncActiveConversationMessages() {
    const conv = getActiveConversation();
    if (!conv) {
      return;
    }
    conv.messages = chatLog.slice();
    conv.updatedAt = new Date().toISOString();
    saveConversationStore();
  }

  function persistConversation() {
    syncActiveConversationMessages();
    renderChatList();
    updateChatHeadingFromActive();
  }

  function updateChatHeadingFromActive() {
    const conv = getActiveConversation();
    chatHeading.textContent = conv && conv.title ? conv.title : "Virtual reference desk";
  }

  function renderMessagesFromLog() {
    messageList.innerHTML = "";
    chatLog.forEach(function (entry) {
      if (!entry || !entry.role || !entry.text) {
        return;
      }
      appendMessage(entry.role, entry.text, { skipPersist: true });
    });
  }

  function loadConversationIntoUI(conv) {
    sessionId = conv.id;
    conversationStore.activeId = conv.id;
    chatLog = conv.messages.map(function (m) {
      return { role: m.role, text: m.text };
    });
    renderMessagesFromLog();
    updateChatHeadingFromActive();
    saveConversationStore();
    renderChatList();
    
    // Restore or reset context usage
    if (contextBarStats && contextBarFill) {
      if (conv.tokensUsed !== undefined && conv.tokensMax !== undefined) {
        const pct = (conv.tokensUsed / conv.tokensMax) * 100;
        contextBarFill.style.width = Math.min(100, Math.max(0, pct)) + "%";
        contextBarFill.className = "context-bar-fill";
        if (pct >= 95) contextBarFill.classList.add("level-pulse");
        else if (pct >= 85) contextBarFill.classList.add("level-red");
        else if (pct >= 60) contextBarFill.classList.add("level-amber");
        contextBarStats.textContent = `${conv.tokensUsed} / ${conv.tokensMax} tokens (${Math.round(pct)}%)`;
      } else {
        contextBarFill.style.width = "0%";
        contextBarFill.className = "context-bar-fill";
        const maxTokens = libraryInfo ? libraryInfo.max_context_tokens : 131072;
        contextBarStats.textContent = `0 / ${maxTokens} tokens (0%)`;
      }
    }
    
    input.focus();
    updateCharCount();
  }

  function renderChatList() {
    if (!chatListEl) {
      return;
    }
    chatListEl.innerHTML = "";
    const sorted = conversationStore.conversations.slice().sort(function (a, b) {
      return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
    });

    sorted.forEach(function (conv) {
      const li = document.createElement("li");
      li.className = "chat-list-item";
      if (conv.id === conversationStore.activeId) {
        li.classList.add("chat-list-item-active");
      }

      const titleText = conv.title || DEFAULT_CHAT_TITLE;
      const lastActiveIso = conv.updatedAt || conv.createdAt || "";

      const selectBtn = document.createElement("button");
      selectBtn.type = "button";
      selectBtn.className = "chat-list-select";
      selectBtn.title = titleText;

      const titleSpan = document.createElement("span");
      titleSpan.className = "chat-list-title";
      titleSpan.textContent = titleText;

      const lastActive = document.createElement("time");
      lastActive.className = "chat-list-last-active";
      lastActive.dateTime = lastActiveIso;
      lastActive.textContent = formatChatLastActive(lastActiveIso);

      selectBtn.appendChild(titleSpan);
      selectBtn.appendChild(lastActive);
      selectBtn.addEventListener("click", function () {
        closeAllChatMenus();
        switchToChat(conv.id);
      });

      const menuWrap = document.createElement("div");
      menuWrap.className = "chat-list-menu-wrap";

      const menuBtn = document.createElement("button");
      menuBtn.type = "button";
      menuBtn.className = "chat-list-menu-btn";
      menuBtn.setAttribute("aria-label", "Chat options");
      menuBtn.setAttribute("aria-expanded", "false");
      menuBtn.textContent = "⋯";

      const popover = document.createElement("div");
      popover.className = "chat-list-menu-popover";
      popover.hidden = true;

      const renameItem = document.createElement("button");
      renameItem.type = "button";
      renameItem.className = "chat-list-menu-item";
      renameItem.textContent = "Rename";
      renameItem.addEventListener("click", function (e) {
        e.stopPropagation();
        closeAllChatMenus();
        renameChat(conv.id);
      });

      const deleteItem = document.createElement("button");
      deleteItem.type = "button";
      deleteItem.className = "chat-list-menu-item chat-list-menu-item-danger";
      deleteItem.textContent = "Delete";
      deleteItem.addEventListener("click", function (e) {
        e.stopPropagation();
        closeAllChatMenus();
        deleteChat(conv.id);
      });

      popover.appendChild(renameItem);
      popover.appendChild(deleteItem);

      menuBtn.addEventListener("click", function (e) {
        e.stopPropagation();
        const willOpen = popover.hidden;
        closeAllChatMenus();
        if (willOpen) {
          popover.hidden = false;
          menuBtn.setAttribute("aria-expanded", "true");
          openChatMenuEl = popover;
        }
      });

      menuWrap.appendChild(menuBtn);
      menuWrap.appendChild(popover);
      li.appendChild(selectBtn);
      li.appendChild(menuWrap);
      chatListEl.appendChild(li);
    });
  }

  function appendMessage(role, text, options) {
    options = options || {};
    const div = document.createElement("article");
    div.className = "message " + (options.extraClass || "message-" + role);

    const meta = document.createElement("div");
    meta.className = "message-meta";

    if (role === "assistant") {
      const avatar = document.createElement("span");
      avatar.className = "message-avatar";
      avatar.textContent = "VD";
      avatar.setAttribute("aria-hidden", "true");
      meta.appendChild(avatar);
    }

    const label = document.createElement("span");
    label.className = "message-role";
    if (role === "user") {
      label.textContent = "You";
    } else if (role === "assistant") {
      label.textContent = libraryInfo ? libraryInfo.assistant_name : "NUCES Virtual Desk";
    } else {
      label.textContent = "System";
    }
    meta.appendChild(label);

    const time = document.createElement("time");
    time.className = "message-time";
    time.dateTime = new Date().toISOString();
    time.textContent = formatTime(new Date());
    meta.appendChild(time);

    div.appendChild(meta);

    const body = document.createElement("p");
    body.className = "message-body";
    if (options.streaming) {
      body.classList.add("streaming");
    }
    body.textContent = text;
    div.appendChild(body);

    messageList.appendChild(div);
    messageList.scrollTop = messageList.scrollHeight;

    if (!options.skipPersist && text && (role === "user" || role === "assistant")) {
      chatLog.push({ role: role, text: text });
      persistConversation();
    }

    // Insert citations if provided
    if (options.ragSources && options.ragSources.length > 0) {
      const details = document.createElement("details");
      details.className = "rag-sources";
      const summary = document.createElement("summary");
      summary.className = "rag-sources-summary";
      summary.textContent = `Sources (${options.ragSources.length})`;
      details.appendChild(summary);
      
      const list = document.createElement("ul");
      list.className = "rag-sources-list";
      options.ragSources.forEach(src => {
        const li = document.createElement("li");
        li.className = "rag-source-item";
        
        const titleSpan = document.createElement("span");
        titleSpan.className = "rag-source-title";
        titleSpan.textContent = src.title;
        
        const simSpan = document.createElement("span");
        simSpan.className = "rag-source-sim";
        simSpan.textContent = Math.round(src.similarity * 100) + "% match";
        
        li.appendChild(titleSpan);
        li.appendChild(simSpan);
        list.appendChild(li);
      });
      details.appendChild(list);
      div.appendChild(details);
      
      // Auto-scroll after adding citations
      setTimeout(() => {
        messageList.scrollTop = messageList.scrollHeight;
      }, 50);
    }

    return body;
  }

  function maybeAutoTitleFromFirstMessage(userText) {
    const conv = getActiveConversation();
    if (!conv || conv.titleManual) {
      return;
    }
    const hasUser = conv.messages.some(function (m) {
      return m.role === "user";
    });
    if (!hasUser) {
      conv.title = deriveChatTitle(userText);
    }
  }

  function clearStreamIdleTimer() {
    if (streamIdleTimer) {
      clearTimeout(streamIdleTimer);
      streamIdleTimer = null;
    }
  }

  function scheduleStreamIdleFinalize() {
    clearStreamIdleTimer();
    streamIdleTimer = setTimeout(function () {
      if (streaming && streamingAssistantBody && streamingAssistantBody.textContent.trim()) {
        finishStreaming();
      }
    }, STREAM_IDLE_MS);
  }

  function clearReplyTimer() {
    if (replyTimer) {
      clearTimeout(replyTimer);
      replyTimer = null;
    }
  }

  function resetUiBusyState() {
    clearReplyTimer();
    clearStreamIdleTimer();
    streaming = false;
    inflightSessionId = null;
    streamingAssistantBody = null;
    streamingMessageEl = null;
    setThinking(false);
    setStreamingControls(false);
    updateCharCount();
  }

  function finishStreaming() {
    clearStreamIdleTimer();
    if (streamingAssistantBody) {
      streamingAssistantBody.classList.remove("streaming");
      const finalText = streamingAssistantBody.textContent.trim();
      if (!finalText) {
        if (streamingMessageEl) {
          streamingMessageEl.remove();
        }
      } else {
        const last = chatLog[chatLog.length - 1];
        if (!last || last.role !== "assistant" || last.text !== finalText) {
          chatLog.push({ role: "assistant", text: finalText });
          persistConversation();
        }
      }
    }
    resetUiBusyState();
    input.focus();
  }

  function handleStreamInterrupted(reason) {
    if (streamingAssistantBody && streamingAssistantBody.textContent.trim()) {
      streamingAssistantBody.classList.remove("streaming");
      appendMessage("system", reason + " Partial reply kept above.", { extraClass: "message-system" });
    } else if (streamingMessageEl) {
      streamingMessageEl.remove();
      appendMessage("error", reason, { extraClass: "message-error" });
    } else if (streaming) {
      appendMessage("error", reason, { extraClass: "message-error" });
    }
    finishStreaming();
  }

  function belongsToInflight(data) {
    if (!data.session_id) {
      return true;
    }
    return data.session_id === inflightSessionId || data.session_id === sessionId;
  }

  function bindStarters() {
    starterPrompts.innerHTML = "";
    STARTERS.forEach(function (text) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "starter-btn";
      btn.textContent = text;
      btn.addEventListener("click", function () {
        if (streaming) {
          return;
        }
        if (startersDisclosure) {
          startersDisclosure.open = false;
        }
        input.value = text;
        updateCharCount();
        form.requestSubmit();
      });
      starterPrompts.appendChild(btn);
    });
    updateStarterDisabled();
  }

  function updateStarterDisabled() {
    const disabled = streaming || !socket || socket.readyState !== WebSocket.OPEN;
    starterPrompts.querySelectorAll(".starter-btn").forEach(function (btn) {
      btn.disabled = disabled;
    });
  }

  async function loadLibraryInfo() {
    try {
      const res = await fetch("/api/library-info");
      if (!res.ok) {
        throw new Error("library info failed");
      }
      libraryInfo = await res.json();
      document.getElementById("sidebar-tagline").textContent = libraryInfo.tagline;
      document.getElementById("sidebar-patron-card").textContent = libraryInfo.sample_patron_card;
      if (composerDisclaimer && libraryInfo.assistant_name) {
        composerDisclaimer.textContent =
          libraryInfo.assistant_name + " can make mistakes so cross check first.";
      }
      if (contextBarStats && libraryInfo.max_context_tokens) {
        const hasSession = conversationStore.activeId && chatLog.length > 0;
        if (!hasSession || !contextBarStats.textContent.includes("/")) {
          contextBarStats.textContent = `0 / ${libraryInfo.max_context_tokens} tokens (0%)`;
        }
      }
    } catch {
      setBanner("Could not load library info. Using default desk settings.", "warn");
    }
  }

  async function loadHealth() {
    try {
      const res = await fetch("/api/health");
      if (!res.ok) {
        throw new Error("health failed");
      }
      const data = await res.json();
      modelHealthy = data.status === "ok";
      modelStatus.textContent = "Model: " + data.model + " (" + data.status + ")";
      modelStatus.classList.toggle("model-pill-ok", modelHealthy);
      modelStatus.classList.toggle("model-pill-bad", !modelHealthy);
      if (!modelHealthy) {
        setBanner("Ollama is not ready. Run: brew services start ollama", "error");
      } else if (socket && socket.readyState === WebSocket.OPEN && !streaming) {
        setBanner("", "");
      }
    } catch {
      modelHealthy = false;
      modelStatus.textContent = "Model: unavailable";
      modelStatus.classList.add("model-pill-bad");
      setBanner("Cannot reach model health endpoint.", "error");
    }
    updateCharCount();
  }

  function scheduleReconnect() {
    if (manualDisconnect || reconnectTimer) {
      return;
    }
    reconnectTimer = setTimeout(function () {
      reconnectTimer = null;
      connect();
    }, 2500);
  }

  function onSocketMessage(event) {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch {
      appendMessage("error", "Invalid server message.", { extraClass: "message-error" });
      finishStreaming();
      return;
    }

    // ---- Assignment 3: voice + tool events (all additive) ----
    if (data.type === "transcript_partial") {
      if (voice.isRecording()) showPartial("… " + data.text);
      return;
    }

    if (data.type === "transcript_final") {
      // Server detected end of speech. From here the turn is exactly like a typed one.
      voice.stopMic(false);
      showPartial("");
      beginTurn(data.text);
      return;
    }

    if (data.type === "audio_chunk") {
      if (data.session_id !== sessionId) return;
      if (speakToggle.checked) {
        stopAudioBtn.hidden = false;
        voice.playChunk(data);
      }
      return;
    }

    if (data.type === "tts_done") {
      return;
    }

    if (data.type === "metrics") {
      showMetrics(data);
      return;
    }

    if (data.type === "tool_call") {
      if (streaming) {
        thinkingLabel.textContent = "Using " + data.name + "…";
        thinkingEl.hidden = false;
      }
      return;
    }

    if (data.type === "tool_result") {
      if (streaming) thinkingLabel.textContent = assistantThinkingLabel();
      return;
    }

    if (data.type === "error" && (voice.isRecording() || !partialTranscript.hidden)) {
      // ASR problem (e.g. nothing heard) before a turn started.
      voice.stopMic(false);
      showPartial("");
      appendMessage("error", data.message || "Voice error.", { extraClass: "message-error" });
      return;
    }

    if (data.type === "token") {
      if (!streaming || !belongsToInflight(data)) {
        return;
      }
      setThinking(false);
      setStreamingControls(true);
      if (!streamingAssistantBody) {
        streamingAssistantBody = appendMessage("assistant", "", { streaming: true });
        streamingMessageEl = streamingAssistantBody.closest(".message");
      }
      streamingAssistantBody.textContent += data.content;
      messageList.scrollTop = messageList.scrollHeight;
      scheduleStreamIdleFinalize();
      return;
    }

    if (data.type === "done") {
      if (streaming) {
        finishStreaming();
      }
      return;
    }

    if (data.type === "error") {
      if (!streaming || !belongsToInflight(data)) {
        return;
      }
      setThinking(false);
      if (streamingMessageEl && (!streamingAssistantBody || !streamingAssistantBody.textContent.trim())) {
        streamingMessageEl.remove();
      } else if (streamingAssistantBody) {
        streamingAssistantBody.classList.remove("streaming");
      }
      appendMessage("error", data.message || "Unknown error.", { extraClass: "message-error" });
      finishStreaming();
      return;
    }

    if (data.type === "context_usage") {
      if (!belongsToInflight(data) && data.session_id !== conversationStore.activeId) return;
      
      const conv = conversationStore.conversations.find(function(c) { return c.id === data.session_id; });
      if (conv) {
        conv.tokensUsed = data.tokens_used;
        conv.tokensMax = data.tokens_max;
        saveConversationStore();
      }

      const pct = (data.tokens_used / data.tokens_max) * 100;
      if (contextBarFill) {
        contextBarFill.style.width = Math.min(100, Math.max(0, pct)) + "%";
        contextBarFill.className = "context-bar-fill";
        if (pct >= 95) contextBarFill.classList.add("level-pulse");
        else if (pct >= 85) contextBarFill.classList.add("level-red");
        else if (pct >= 60) contextBarFill.classList.add("level-amber");
      }
      if (contextBarStats) {
        contextBarStats.textContent = `${data.tokens_used} / ${data.tokens_max} tokens (${Math.round(pct)}%)`;
      }
      return;
    }

    if (data.type === "rag_sources") {
      if (!belongsToInflight(data)) return;
      if (streamingMessageEl) {
        const article = streamingMessageEl.closest(".message");
        if (article) {
          const details = document.createElement("details");
          details.className = "rag-sources";
          const summary = document.createElement("summary");
          summary.className = "rag-sources-summary";
          summary.textContent = `Sources (${data.sources.length})`;
          details.appendChild(summary);
          
          const list = document.createElement("ul");
          list.className = "rag-sources-list";
          data.sources.forEach(src => {
            const li = document.createElement("li");
            li.className = "rag-source-item";
            
            const titleSpan = document.createElement("span");
            titleSpan.className = "rag-source-title";
            titleSpan.textContent = src.title;
            
            const simSpan = document.createElement("span");
            simSpan.className = "rag-source-sim";
            simSpan.textContent = Math.round(src.similarity * 100) + "% match";
            
            li.appendChild(titleSpan);
            li.appendChild(simSpan);
            list.appendChild(li);
          });
          details.appendChild(list);
          article.appendChild(details);
          messageList.scrollTop = messageList.scrollHeight;
        }
      }
      return;
    }

    if (data.type === "summarise_done") {
      if (btnSummarise) {
        btnSummarise.classList.remove("loading");
        btnSummarise.disabled = false;
      }
      showToast(data.message);
      const pct = (data.tokens_used / data.tokens_max) * 100;
      if (contextBarFill) {
        contextBarFill.style.width = Math.min(100, Math.max(0, pct)) + "%";
        contextBarFill.className = "context-bar-fill";
        if (pct >= 95) contextBarFill.classList.add("level-pulse");
        else if (pct >= 85) contextBarFill.classList.add("level-red");
        else if (pct >= 60) contextBarFill.classList.add("level-amber");
      }
      if (contextBarStats) {
        contextBarStats.textContent = `${data.tokens_used} / ${data.tokens_max} tokens (${Math.round(pct)}%)`;
      }
      return;
    }

    if (data.type === "session_reset") {
      return;
    }
  }

  function connect() {
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
      return;
    }
    manualDisconnect = false;
    setStatus("connecting");
    socket = new WebSocket(WS_PATH);
    socket.addEventListener("open", function () {
      setStatus("open");
      loadHealth();
      updateStarterDisabled();
      updateCharCount();
    });
    socket.addEventListener("close", function () {
      setStatus("closed");
      scheduleReconnect();
      updateStarterDisabled();
      updateCharCount();
    });
    socket.addEventListener("error", function () {
      setStatus("closed");
    });
    socket.addEventListener("message", onSocketMessage);
  }

  function startReplyTimer(requestId) {
    clearReplyTimer();
    replyTimer = setTimeout(function () {
      if (requestId !== activeRequestId) {
        return;
      }
      handleStreamInterrupted("The library model took too long to respond. Please try again.");
    }, REPLY_TIMEOUT_MS);
  }

  // Shared by typed and spoken turns: show the user bubble and wait for the reply.
  function beginTurn(text) {
    maybeAutoTitleFromFirstMessage(text);

    activeRequestId += 1;
    const requestId = activeRequestId;
    inflightSessionId = sessionId;
    streaming = true;
    streamingAssistantBody = null;
    streamingMessageEl = null;
    setThinking(true);
    setStreamingControls(true);
    appendMessage("user", text);
    startReplyTimer(requestId);
  }

  function sendUserMessage(text) {
    if (!text || text.length > MAX_CHARS) {
      appendMessage("error", "Message must be between 1 and 4000 characters.", { extraClass: "message-error" });
      return;
    }
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      appendMessage("error", "Not connected. Use Reconnect, then try again.", { extraClass: "message-error" });
      connect();
      return;
    }
    if (streaming) {
      return;
    }
    if (!modelHealthy) {
      setBanner("Model is not ready. Replies may fail until Ollama is running.", "warn");
    }

    beginTurn(text);
    socket.send(
      JSON.stringify({
        type: "user_message",
        session_id: sessionId,
        content: text,
      })
    );
  }

  async function createNewChat(force) {
    if (streaming && !force) {
      const ok = await showConfirm({
        title: "Reply in progress",
        message: "Start a new chat before the current reply finishes?",
        confirmLabel: "New chat",
      });
      if (!ok) {
        return;
      }
      resetUiBusyState();
    }
    activeRequestId += 1;
    if (force) {
      resetUiBusyState();
    }
    syncActiveConversationMessages();

    const existingEmpty = findEmptyChat();
    if (existingEmpty) {
      if (existingEmpty.id !== conversationStore.activeId) {
        await switchToChat(existingEmpty.id, true);
      }
      input.focus();
      return;
    }

    const newId = crypto.randomUUID();
    const greeting = defaultGreeting();
    const now = new Date().toISOString();
    const conv = {
      id: newId,
      title: DEFAULT_CHAT_TITLE,
      titleManual: false,
      createdAt: now,
      updatedAt: now,
      messages: [{ role: "assistant", text: greeting }],
    };
    conversationStore.conversations.push(conv);
    conversationStore.activeId = newId;
    saveConversationStore();
    loadConversationIntoUI(conv);
    connectionLine.textContent = "New chat started at " + formatTime(new Date());
  }

  async function switchToChat(id, force) {
    if (id === conversationStore.activeId) {
      return;
    }
    if (streaming && !force) {
      const ok = await showConfirm({
        title: "Reply in progress",
        message: "Switch to another chat before the current reply finishes?",
        confirmLabel: "Switch chat",
      });
      if (!ok) {
        return;
      }
      resetUiBusyState();
    }
    closeAllChatMenus();
    syncActiveConversationMessages();
    const conv = conversationStore.conversations.find(function (c) {
      return c.id === id;
    });
    if (!conv) {
      return;
    }
    loadConversationIntoUI(conv);
    connectionLine.textContent = "Secure session with the virtual reference desk";
  }

  async function renameChat(id) {
    const conv = conversationStore.conversations.find(function (c) {
      return c.id === id;
    });
    if (!conv) {
      return;
    }
    const next = await showPrompt({
      title: "Rename chat",
      message: "Choose a short label for this conversation.",
      defaultValue: conv.title || DEFAULT_CHAT_TITLE,
      confirmLabel: "Save",
    });
    if (next === null) {
      return;
    }
    const trimmed = next.trim();
    if (!trimmed) {
      return;
    }
    conv.title = trimmed.slice(0, 80);
    conv.titleManual = true;
    conv.updatedAt = new Date().toISOString();
    saveConversationStore();
    renderChatList();
    if (conv.id === conversationStore.activeId) {
      updateChatHeadingFromActive();
    }
  }

  async function deleteChat(id) {
    const conv = conversationStore.conversations.find(function (c) {
      return c.id === id;
    });
    if (!conv) {
      return;
    }
    const label = conv.title || DEFAULT_CHAT_TITLE;
    const ok = await showConfirm({
      title: "Delete chat",
      message: 'Remove "' + label + '" from this browser? This cannot be undone.',
      confirmLabel: "Delete",
    });
    if (!ok) {
      return;
    }

    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(
        JSON.stringify({
          type: "reset_session",
          session_id: id,
        })
      );
    }

    conversationStore.conversations = conversationStore.conversations.filter(function (c) {
      return c.id !== id;
    });

    if (conversationStore.activeId === id) {
      if (conversationStore.conversations.length === 0) {
        createNewChat(true);
        return;
      }
      const sorted = conversationStore.conversations.slice().sort(function (a, b) {
        return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
      });
      loadConversationIntoUI(sorted[0]);
    } else {
      saveConversationStore();
      renderChatList();
    }
  }

  function initConversations() {
    const loaded = loadConversationStore();
    if (!loaded) {
      migrateLegacySessionStorage();
    }
    if (conversationStore.conversations.length === 0) {
      createNewChat(true);
      return;
    }
    const active = conversationStore.conversations.find(function (c) {
      return c.id === conversationStore.activeId;
    });
    if (active) {
      loadConversationIntoUI(active);
    } else {
      const sorted = conversationStore.conversations.slice().sort(function (a, b) {
        return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
      });
      loadConversationIntoUI(sorted[0]);
    }
    connectionLine.textContent = "Secure session with the virtual reference desk";
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    const text = input.value.trim();
    if (!text || streaming) {
      return;
    }
    input.value = "";
    updateCharCount();
    sendUserMessage(text);
  });

  micBtn.addEventListener("click", function () {
    if (voice.isRecording()) {
      voice.stopMic(true); // user pressed Stop: server finalises what it has
    } else if (!streaming) {
      voice.stopPlayback();
      voice.startMic(speakToggle.checked);
    }
  });

  stopAudioBtn.addEventListener("click", function () {
    voice.stopPlayback();
    stopAudioBtn.hidden = true;
  });

  speakToggle.addEventListener("change", function () {
    if (!speakToggle.checked) {
      voice.stopPlayback();
      stopAudioBtn.hidden = true;
    }
  });

  // Hide the Stop button once the queued reply audio has finished playing.
  setInterval(function () {
    if (!stopAudioBtn.hidden && !voice.isPlaying()) stopAudioBtn.hidden = true;
  }, 500);

  newSessionBtn.addEventListener("click", function () {
    createNewChat(false);
  });

  if (btnSummarise) {
    btnSummarise.addEventListener("click", function () {
      if (streaming || !socket || socket.readyState !== WebSocket.OPEN) return;
      btnSummarise.classList.add("loading");
      btnSummarise.disabled = true;
      socket.send(JSON.stringify({
        type: "request_summarise",
        session_id: conversationStore.activeId
      }));
    });
  }

  if (newChatSidebarBtn) {
    newChatSidebarBtn.addEventListener("click", function () {
      createNewChat(false);
    });
  }

  retryBtn.addEventListener("click", function () {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    connect();
  });

  input.addEventListener("input", updateCharCount);

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  window.addEventListener("beforeunload", function () {
    syncActiveConversationMessages();
    manualDisconnect = true;
    if (socket) {
      socket.close();
    }
  });

  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "hidden") {
      syncActiveConversationMessages();
    }
  });

  if (btnViewSources) {
    btnViewSources.addEventListener("click", async function () {
      sourcesModal.hidden = false;
      sourceTitleDisplay.textContent = "Loading sources...";
      sourceContentDisplay.textContent = "";
      sourcesList.innerHTML = "";
      try {
        const res = await fetch("/api/sources");
        const data = await res.json();
        if (data && data.length) {
          sourceTitleDisplay.textContent = "Select a source from the left";
          data.forEach(function (src) {
            const li = document.createElement("li");
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "btn btn-outline";
            btn.style.width = "100%";
            btn.style.textAlign = "left";
            btn.textContent = src.title;
            btn.addEventListener("click", async function () {
              sourceTitleDisplay.textContent = "Loading " + src.title + "...";
              sourceContentDisplay.textContent = "";
              const docRes = await fetch("/api/sources/" + src.doc_id);
              const docData = await docRes.json();
              if (docData && docData.content) {
                sourceTitleDisplay.textContent = src.title;
                sourceContentDisplay.textContent = docData.content;
              } else {
                sourceTitleDisplay.textContent = "Error loading source";
              }
            });
            li.appendChild(btn);
            sourcesList.appendChild(li);
          });
        } else {
          sourceTitleDisplay.textContent = "No sources found";
        }
      } catch (err) {
        sourceTitleDisplay.textContent = "Failed to load sources";
      }
    });
  }

  if (btnCloseSourcesModal) {
    btnCloseSourcesModal.addEventListener("click", function () {
      sourcesModal.hidden = true;
    });
  }

  if (sourcesModalBackdrop) {
    sourcesModalBackdrop.addEventListener("click", function () {
      sourcesModal.hidden = true;
    });
  }

  resetUiBusyState();
  bindModalControls();
  bindStarters();
  updateCharCount();
  loadLibraryInfo().then(function () {
    connect();
    initConversations();
  });
  loadHealth();
  healthTimer = setInterval(loadHealth, HEALTH_POLL_MS);
})();
