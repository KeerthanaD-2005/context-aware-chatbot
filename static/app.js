// DOM Elements
const messagesEl = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const errorEl = document.getElementById("error");
const errorText = document.getElementById("error-text");
const errorDismiss = document.getElementById("error-dismiss");
const newChatButton = document.getElementById("new-chat");
const clearButton = document.getElementById("clear-btn");
const strategySelect = document.getElementById("strategy-select");
const charCounter = document.getElementById("char-counter");

// Context Inspector DOM elements
const statTotalTurns = document.getElementById("stat-total-turns");
const statActiveMsgs = document.getElementById("stat-active-msgs");
const statTruncated = document.getElementById("stat-truncated");
const contextStrategyBadge = document.getElementById("context-strategy-badge");
const displayConversationId = document.getElementById("display-conversation-id");
const providerBadge = document.getElementById("provider-badge");
const topbarModelLabel = document.getElementById("topbar-model-label");

let conversationId = crypto.randomUUID();
let currentTurnCount = 0;

// Initialize app & fetch backend health/config
async function initApp() {
  updateSessionDisplay();
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      if (data.provider !== "none") {
        providerBadge.textContent = `${data.provider.toUpperCase()} (${data.model})`;
        providerBadge.style.background = "#dcfce7";
        providerBadge.style.color = "#15803d";
        topbarModelLabel.textContent = `${data.provider.toUpperCase()} · ${data.model}`;
      } else {
        providerBadge.textContent = "No API Key configured";
        providerBadge.style.background = "#fee2e2";
        providerBadge.style.color = "#b91c1c";
        topbarModelLabel.textContent = "API KEY MISSING";
      }
      if (data.context_strategy && strategySelect) {
        strategySelect.value = data.context_strategy;
        updateStrategyBadge(data.context_strategy);
      }
    }
  } catch (err) {
    console.warn("Could not fetch server health:", err);
  }
}

function updateSessionDisplay() {
  if (displayConversationId) {
    displayConversationId.textContent = conversationId.slice(0, 8) + "…";
  }
}

function updateStrategyBadge(strategy) {
  if (!contextStrategyBadge) return;
  if (strategy === "summarization") {
    contextStrategyBadge.textContent = "Summarization";
    contextStrategyBadge.style.background = "#fef3c7";
    contextStrategyBadge.style.color = "#b45309";
  } else {
    contextStrategyBadge.textContent = "Sliding Window";
    contextStrategyBadge.style.background = "#eef2ff";
    contextStrategyBadge.style.color = "#4f46e5";
  }
}

function hideWelcome() {
  const welcome = document.getElementById("welcome");
  if (welcome) welcome.remove();
}

/**
 * Basic markdown-style parser for safe formatting:
 * Supports ```code blocks```, `inline code`, **bold**, *italic*, and line breaks.
 */
function renderFormattedText(text) {
  const container = document.createElement("div");
  container.className = "message-body";

  // Check for code fences ```
  const parts = text.split(/(```[\s\S]*?```)/g);

  for (const part of parts) {
    if (part.startsWith("```") && part.endsWith("```")) {
      const rawCode = part.slice(3, -3);
      // Remove optional language identifier on the first line
      const lines = rawCode.split("\n");
      let codeContent = rawCode;
      if (lines.length > 1 && !lines[0].includes(" ")) {
        codeContent = lines.slice(1).join("\n");
      }
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.textContent = codeContent.trim();
      pre.appendChild(code);
      container.appendChild(pre);
    } else if (part.trim()) {
      // Process paragraphs and simple inline formatting
      const paragraphs = part.split(/\n\s*\n/);
      for (const pText of paragraphs) {
        if (!pText.trim()) continue;

        // Check if list item
        if (pText.trim().startsWith("- ") || pText.trim().startsWith("* ")) {
          const ul = document.createElement("ul");
          const items = pText.trim().split(/\n(?=[- *]\s)/);
          for (const item of items) {
            const li = document.createElement("li");
            li.innerHTML = formatInline(item.replace(/^[- *]\s*/, ""));
            ul.appendChild(li);
          }
          container.appendChild(ul);
        } else {
          const p = document.createElement("p");
          p.innerHTML = formatInline(pText);
          container.appendChild(p);
        }
      }
    }
  }

  return container;
}

function formatInline(str) {
  // Escape raw HTML entities first to prevent XSS
  const safe = str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  return safe
    // `inline code`
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    // **bold**
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    // *italic*
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    // newlines
    .replace(/\n/g, "<br>");
}

function addMessage(role, text) {
  hideWelcome();
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "You" : "✦";

  const content = document.createElement("div");
  content.className = "message-content";

  const roleLabel = document.createElement("div");
  roleLabel.className = "message-role";
  roleLabel.textContent = role === "user" ? "You" : "StudyBuddy";

  const inner = document.createElement("div");
  inner.className = "message-content-inner";

  if (role === "assistant" && text !== "Thinking…") {
    inner.appendChild(renderFormattedText(text));
  } else {
    const textNode = document.createElement("div");
    textNode.textContent = text;
    inner.appendChild(textNode);
  }

  content.append(roleLabel, inner);
  wrapper.append(avatar, content);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrapper;
}

function showError(message) {
  errorText.textContent = message;
  errorEl.hidden = false;
  errorEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function hideError() {
  errorEl.hidden = true;
}

if (errorDismiss) {
  errorDismiss.addEventListener("click", hideError);
}

function setBusy(busy) {
  sendButton.disabled = busy;
  input.disabled = busy;
  sendButton.textContent = busy ? "…" : "↑";
}

function updateContextStats(stats) {
  if (!stats) return;
  if (statTotalTurns) {
    statTotalTurns.textContent = currentTurnCount;
  }
  if (statActiveMsgs) {
    statActiveMsgs.textContent = `${stats.messages_in_context} / ${stats.context_limit}`;
  }
  if (statTruncated) {
    statTruncated.textContent = stats.truncated_messages || 0;
  }
  if (stats.strategy) {
    updateStrategyBadge(stats.strategy);
  }
}

async function sendMessage(text) {
  const clean = text.trim();
  if (!clean) return;

  hideError();
  addMessage("user", clean);
  input.value = "";
  input.style.height = "auto";
  updateCharCounter();
  setBusy(true);

  const typing = addMessage("assistant", "Thinking…");
  typing.querySelector(".message-content-inner div").className = "typing";

  const chosenStrategy = strategySelect ? strategySelect.value : "sliding_window";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: clean,
        conversation_id: conversationId,
        strategy: chosenStrategy,
      }),
    });

    const data = await response.json();
    typing.remove();

    if (!response.ok) {
      throw new Error(data.error || `Request failed with status code ${response.status}`);
    }

    conversationId = data.conversation_id;
    currentTurnCount += 1;
    updateSessionDisplay();

    addMessage("assistant", data.reply);
    updateContextStats(data.context_stats);

    if (data.provider && data.model) {
      topbarModelLabel.textContent = `${data.provider.toUpperCase()} · ${data.model}`;
    }
  } catch (error) {
    typing.remove();
    showError(error.message || "Could not reach the server. Please verify that it is running.");
  } finally {
    setBusy(false);
    input.focus();
  }
}

// Event Listeners
form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!sendButton.disabled) {
    sendMessage(input.value);
  }
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

function updateCharCounter() {
  if (charCounter) {
    charCounter.textContent = `${input.value.length} / 8000`;
  }
}

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
  updateCharCounter();
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    sendMessage(button.textContent.trim());
  });
});

function resetChatUI() {
  conversationId = crypto.randomUUID();
  currentTurnCount = 0;
  updateSessionDisplay();

  if (statTotalTurns) statTotalTurns.textContent = "0";
  if (statActiveMsgs) statActiveMsgs.textContent = "0 / 10";
  if (statTruncated) statTruncated.textContent = "0";

  messagesEl.innerHTML = `
    <div class="welcome" id="welcome">
      <div class="welcome-icon">✦</div>
      <h1>Fresh conversation started</h1>
      <p>This conversation has a fresh, independent context window. Any previous details were stored only in the old session.</p>
      <div class="suggestions">
        <button class="suggestion">My project is called StudyBuddy and it helps students learn Python.</button>
        <button class="suggestion">Explain the concept of APIs using a coffee shop analogy.</button>
      </div>
    </div>`;

  hideError();
  input.value = "";
  input.style.height = "auto";
  updateCharCounter();
  input.focus();

  // Re-attach suggestion listeners
  messagesEl.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
      sendMessage(button.textContent.trim());
    });
  });
}

newChatButton.addEventListener("click", resetChatUI);
if (clearButton) {
  clearButton.addEventListener("click", resetChatUI);
}

if (strategySelect) {
  strategySelect.addEventListener("change", () => {
    updateStrategyBadge(strategySelect.value);
  });
}

// Initialize on page load
initApp();
