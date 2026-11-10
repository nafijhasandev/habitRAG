/**
 * Chat interface logic for Ask Atomic Habits.
 *
 * Handles:
 * - Sending queries to the /api/query endpoint
 * - Parsing Server-Sent Events (SSE) for streaming responses
 * - Rendering messages and sources
 * - Auto-resizing the input textarea
 * - Health check on page load
 */

const chatContainer = document.getElementById("chatContainer");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const welcome = document.getElementById("welcome");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const app = document.querySelector(".app");
const chatList = document.getElementById("chatList");
const historyNotice = document.getElementById("historyNotice");
const sidebarToggle = document.getElementById("sidebarToggle");
const mainPanel = document.querySelector(".main-panel");
const settingsDialog = document.getElementById("settingsDialog");
const showSourcesToggle = document.getElementById("showSourcesToggle");
const settingsNotice = document.getElementById("settingsNotice");
const themeToggle = document.getElementById("themeToggle");
const themeNotice = document.getElementById("themeNotice");
const pdfPanel = document.getElementById("pdfPanel");
const pdfFrame = document.getElementById("pdfFrame");
const pdfTopic = document.getElementById("pdfTopic");
const pdfPage = document.getElementById("pdfPage");
const escapeHtml = AnswerFormat.escapeHtml;

let isStreaming = false;
let chats = [];
try {
    chats = ChatHistory.load(localStorage);
} catch {
    historyNotice.textContent = "Saved chats could not be loaded in this browser.";
    historyNotice.hidden = false;
}
let activeChatId = null;
let settings = { showSources: true, theme: "dark" };
try {
    settings = ChatSettings.load(localStorage);
} catch {
    settingsNotice.textContent = "Saved settings could not be loaded in this browser.";
    settingsNotice.hidden = false;
}

function applySettings() {
    document.documentElement.dataset.theme = settings.theme;
    showSourcesToggle.checked = settings.showSources;
    const nextTheme = settings.theme === "dark" ? "light" : "dark";
    themeToggle.setAttribute("aria-label", `Switch to ${nextTheme} mode`);
    themeToggle.title = `Switch to ${nextTheme} mode`;
    app.classList.toggle("hide-sources", !settings.showSources);
}

function saveSettings() {
    try {
        ChatSettings.save(localStorage, settings);
        settingsNotice.hidden = true;
        themeNotice.hidden = true;
    } catch {
        settingsNotice.textContent = "Settings could not be saved in this browser.";
        settingsNotice.hidden = false;
        themeNotice.textContent = "Preference not saved";
        themeNotice.hidden = false;
    }
}

function saveHistory() {
    try {
        ChatHistory.save(localStorage, chats);
        historyNotice.hidden = true;
    } catch {
        historyNotice.textContent = "Chat history could not be saved in this browser.";
        historyNotice.hidden = false;
    }
}

function currentChat() {
    return chats.find(chat => chat.id === activeChatId);
}

function recordMessage(message) {
    currentChat().messages.push(message);
    saveHistory();
}

function renderChatList() {
    chatList.replaceChildren();
    for (const chat of [...chats].reverse()) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = chat.title;
        button.title = chat.title;
        button.className = chat.id === activeChatId ? "active" : "";
        button.setAttribute("aria-current", chat.id === activeChatId ? "page" : "false");
        button.addEventListener("click", () => {
            if (isStreaming) return;
            activeChatId = chat.id;
            renderConversation();
            renderChatList();
            closeSidebar();
        });
        chatList.appendChild(button);
    }
}

function renderConversation() {
    closePdf();
    chatContainer.replaceChildren();
    const chat = currentChat();
    mainPanel.classList.toggle("is-empty", !chat);
    if (!chat) {
        chatContainer.appendChild(welcome);
        return;
    }
    for (const message of chat.messages) {
        if (message.role === "user") {
            addUserMessage(message.text);
        } else {
            addAssistantMessage(message.text, message.sources || [], message.error);
        }
    }
    scrollToBottom();
}

function syncSidebarState() {
    const mobile = window.matchMedia("(max-width: 640px)").matches;
    const open = mobile ? app.classList.contains("sidebar-open") :
        !app.classList.contains("sidebar-collapsed");
    sidebarToggle.setAttribute("aria-expanded", String(open));
}

function closeSidebar() {
    app.classList.remove("sidebar-open");
    syncSidebarState();
}

function closePdf() {
    pdfPanel.hidden = true;
    app.classList.remove("pdf-open");
    pdfFrame.removeAttribute("src");
}

function openPdf(page, topic) {
    if (!Number.isSafeInteger(page) || page < 1) return;
    pdfPage.textContent = `Page ${page}`;
    pdfTopic.textContent = topic || "Atomic Habits";
    pdfFrame.src = `/book.pdf#page=${page}`;
    pdfPanel.hidden = false;
    app.classList.add("pdf-open");
}

// ===== Health Check =====

async function checkHealth() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();

        if (data.llm_configured && data.index_loaded) {
            statusDot.className = "dot online";
            statusText.textContent = `Ready · ${data.chunk_count} passages`;
        } else if (data.index_loaded) {
            statusDot.className = "dot offline";
            statusText.textContent = "Add an API key to .env";
        } else {
            statusDot.className = "dot offline";
            statusText.textContent = "Index unavailable";
        }
    } catch {
        statusDot.className = "dot offline";
        statusText.textContent = "Server unreachable";
    }
}

// ===== Message Rendering =====

function addUserMessage(text) {
    if (welcome) welcome.remove();

    const msg = document.createElement("div");
    msg.className = "message user";
    msg.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
    chatContainer.appendChild(msg);
    scrollToBottom();
}

function createAssistantMessage() {
    if (welcome) welcome.remove();

    const msg = document.createElement("div");
    msg.className = "message assistant";

    const avatar = document.createElement("img");
    avatar.className = "assistant-avatar";
    avatar.src = "/static/atomic-habits-logo.png";
    avatar.alt = "";

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const loading = document.createElement("div");
    loading.className = "loading-dots";
    loading.innerHTML = "<span></span><span></span><span></span>";

    bubble.appendChild(loading);
    msg.appendChild(avatar);
    msg.appendChild(bubble);
    chatContainer.appendChild(msg);
    scrollToBottom();

    return { msg, bubble, loading };
}

function addAssistantMessage(text, sources, isError) {
    const { bubble, loading } = createAssistantMessage();
    loading.remove();
    const content = document.createElement("div");
    content.className = isError ? "error-msg" : "answer-content";
    content.innerHTML = AnswerFormat.formatHtml(text, sources);
    bubble.appendChild(content);
    renderSources(bubble, sources);
}

function renderSources(bubble, sources) {
    const validSources = (sources || []).filter(source => source && typeof source === "object");
    if (validSources.length === 0) return;

    const container = document.createElement("details");
    container.className = "sources";

    const header = document.createElement("summary");
    header.textContent = `${validSources.length} sources`;
    container.appendChild(header);

    validSources.forEach(source => {
        const meta = source.metadata && typeof source.metadata === "object" ? source.metadata : {};
        const chapter = String(meta.chapter || "").trim();
        const pageStart = meta.page_start || "?";
        const pageEnd = meta.page_end || "?";
        const preview = source.text_preview || "";
        const chapterLabel = /^\d+$/.test(chapter) ? `Chapter ${chapter}` : chapter;
        const pdfPage = AnswerFormat.sourcePage(source);
        const topic = String(meta.section || "").trim();
        const item = document.createElement("div");
        item.className = "source-item";
        item.innerHTML = `
            <div class="source-meta">
                <strong>Atomic Habits</strong>
                ${chapterLabel ? `<strong>${escapeHtml(chapterLabel)}</strong>` : ""}
                ${topic && topic !== chapterLabel ? `<strong>${escapeHtml(topic)}</strong>` : ""}
                <span>Pages ${escapeHtml(pageStart)}-${escapeHtml(pageEnd)}</span>
                ${pdfPage ? `<a class="source-link" ${AnswerFormat.sourceLink(source)}>View topic on page ${pdfPage}</a>` : ""}
            </div>
            <div class="source-text">${escapeHtml(preview)}...</div>
        `;
        container.appendChild(item);
    });

    bubble.appendChild(container);
}

// ===== Streaming Query =====

async function sendQuery(question) {
    if (isStreaming || !question.trim()) return;

    isStreaming = true;
    sendBtn.disabled = true;
    mainPanel.classList.remove("is-empty");
    if (!activeChatId) {
        const chat = ChatHistory.create(crypto.randomUUID(), question);
        chats.push(chat);
        activeChatId = chat.id;
        renderChatList();
    }
    addUserMessage(question);
    recordMessage({ role: "user", text: question });

    const { bubble, loading } = createAssistantMessage();

    let answerDiv = null;
    let sources = null;
    let answerText = "";
    let errorText = "";
    let requestError = "The response could not be completed. Please try again.";

    try {
        const response = await fetch("/api/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        if (!response.ok) {
            if (response.status === 422) {
                requestError = "Please enter a question up to 1000 characters.";
            }
            throw new Error("Request failed");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
                const event = StreamEvents.parse(line);
                if (!event) continue;

                if (event.type === "sources") {
                    sources = event.sources;
                }

                if (event.type === "chunk") {
                    if (!answerDiv) {
                        loading.remove();
                        answerDiv = document.createElement("div");
                        answerDiv.className = "answer-content";
                        bubble.appendChild(answerDiv);
                    }
                    answerDiv.textContent += event.content;
                    answerText += event.content;
                    scrollToBottom();
                }

                if (event.type === "error") {
                    loading.remove();
                    const errDiv = document.createElement("div");
                    errDiv.className = "error-msg";
                    errDiv.textContent = event.message;
                    bubble.appendChild(errDiv);
                    errorText = event.message;
                }

                if (event.type === "done") {
                    loading.remove();
                    if (typeof event.answer === "string" && event.answer) {
                        answerText = event.answer;
                    }
                    if (answerDiv) {
                        answerDiv.innerHTML = AnswerFormat.formatHtml(answerText, sources);
                    }
                    renderSources(bubble, sources);
                }
            }
        }
    } catch {
        loading.remove();
        const errDiv = document.createElement("div");
        errDiv.className = "error-msg";
        errDiv.textContent = requestError;
        bubble.appendChild(errDiv);
        errorText = requestError;
    } finally {
        if (!answerText && !errorText) {
            errorText = "The response ended unexpectedly. Please try again.";
            loading.remove();
            const errDiv = document.createElement("div");
            errDiv.className = "error-msg";
            errDiv.textContent = errorText;
            bubble.appendChild(errDiv);
        }
        recordMessage({
            role: "assistant",
            text: [answerText, errorText].filter(Boolean).join("\n\n"),
            sources: answerText ? sources || [] : [],
            error: !answerText,
        });
        isStreaming = false;
        sendBtn.disabled = false;
        queryInput.focus();
        scrollToBottom();
    }
}

// ===== Event Handlers =====

function handleSubmit(e) {
    e.preventDefault();
    const question = queryInput.value.trim();
    if (question) {
        queryInput.value = "";
        queryInput.style.height = "auto";
        sendQuery(question);
    }
}

function askSuggestion(btn) {
    sendQuery(btn.textContent);
}

document.getElementById("newChatBtn").addEventListener("click", () => {
    if (isStreaming) return;
    activeChatId = null;
    renderConversation();
    renderChatList();
    closeSidebar();
    queryInput.focus();
});

chatContainer.addEventListener("click", event => {
    const link = event.target.closest("a[data-pdf-page]");
    if (!link || !chatContainer.contains(link)) return;
    event.preventDefault();
    openPdf(Number(link.dataset.pdfPage), link.dataset.pdfTopic);
});
document.getElementById("closePdfBtn").addEventListener("click", closePdf);

sidebarToggle.addEventListener("click", () => {
    if (window.matchMedia("(max-width: 640px)").matches) {
        app.classList.toggle("sidebar-open");
    } else {
        app.classList.toggle("sidebar-collapsed");
    }
    syncSidebarState();
});

document.getElementById("sidebarBackdrop").addEventListener("click", closeSidebar);
window.addEventListener("resize", syncSidebarState);

document.getElementById("settingsBtn").addEventListener("click", () => {
    closeSidebar();
    settingsDialog.showModal();
});
document.getElementById("closeSettingsBtn").addEventListener("click", () => {
    settingsDialog.close();
});
showSourcesToggle.addEventListener("change", () => {
    settings = { ...settings, showSources: showSourcesToggle.checked };
    applySettings();
    saveSettings();
});
themeToggle.addEventListener("click", () => {
    settings = { ...settings, theme: settings.theme === "dark" ? "light" : "dark" };
    applySettings();
    saveSettings();
});

// Auto-resize textarea
queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + "px";
});

// Enter to send, Shift+Enter for newline
queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit(e);
    }
});

// ===== Utilities =====

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ===== Init =====
renderChatList();
renderConversation();
applySettings();
syncSidebarState();
checkHealth();
