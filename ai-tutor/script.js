(function () {
  "use strict";

  // Theme handling
  const htmlElement = document.documentElement;
  const themeToggleButton = document.getElementById("themeToggle");
  const storedTheme = localStorage.getItem("theme");
  if (storedTheme === "light" || storedTheme === "dark") {
    htmlElement.setAttribute("data-theme", storedTheme);
  } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
    htmlElement.setAttribute("data-theme", "light");
  }
  themeToggleButton?.addEventListener("click", () => {
    const newTheme = htmlElement.getAttribute("data-theme") === "light" ? "dark" : "light";
    htmlElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("theme", newTheme);
  });

  // Footer year
  const yearSpan = document.getElementById("year");
  if (yearSpan) yearSpan.textContent = String(new Date().getFullYear());

  // Smooth internal nav
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (e) => {
      const href = anchor.getAttribute("href");
      if (!href) return;
      const targetId = href.slice(1);
      const el = document.getElementById(targetId);
      if (el) {
        e.preventDefault();
        el.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });

  // Chat demo
  const chatWindow = document.getElementById("chatWindow");
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");

  /**
   * Append a message to the chat window
   * @param {string} role - "user" | "ai"
   * @param {string} text
   */
  function appendMessage(role, text) {
    if (!chatWindow) return;
    const wrapper = document.createElement("div");
    wrapper.className = `msg ${role === "user" ? "msg--user" : "msg--ai"}`;
    wrapper.innerHTML = `<div>${escapeHtml(text)}</div><span class="msg__meta">${role === "user" ? "You" : "AI Tutor"}</span>`;
    chatWindow.appendChild(wrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    clampHistory();
  }

  function clampHistory() {
    if (!chatWindow) return;
    const nodes = Array.from(chatWindow.children);
    const maxMessages = 100;
    if (nodes.length > maxMessages) {
      nodes.slice(0, nodes.length - maxMessages).forEach((n) => n.remove());
    }
  }

  function escapeHtml(unsafe) {
    return unsafe
      .replaceAll(/&/g, "&amp;")
      .replaceAll(/</g, "&lt;")
      .replaceAll(/>/g, "&gt;")
      .replaceAll(/\"/g, "&quot;")
      .replaceAll(/'/g, "&#039;");
  }

  function think(text) {
    // Simple, deterministic simulated tutor response
    const trimmed = text.trim();
    if (!trimmed) return "Could you share your question?";
    const starters = [
      "Great question!",
      "Let's break it down:",
      "Here's a simple way to see it:",
      "Think of it like this:",
    ];
    const examples = [
      "Example:",
      "Quick check:",
      "Try this:",
      "Analogy:",
    ];
    const starter = starters[trimmed.length % starters.length];
    const example = examples[(trimmed.length >> 1) % examples.length];
    const summary = "Key idea: focus on the relationship between inputs and outputs.";
    return [
      `${starter} ${explainBrief(trimmed)}`,
      `\n\n${example} ${makeExample(trimmed)}`,
      `\n\n${summary}`,
    ].join("");
  }

  function explainBrief(q) {
    if (/derivative|calculus|dx|dy/i.test(q)) return "The derivative measures how fast a function changes at a point.";
    if (/probab|bayes|likeli/i.test(q)) return "Probability quantifies uncertainty; Bayes updates beliefs with new evidence.";
    if (/matrix|vector|linear/i.test(q)) return "Linear algebra studies vectors and transformations that preserve linearity.";
    if (/python|code|algorithm/i.test(q)) return "Start with a clear problem, define inputs/outputs, then iterate on a solution.";
    return "Let's identify what is given, what is asked, and connect them step by step.";
  }

  function makeExample(q) {
    if (/derivative|calculus|dx|dy/i.test(q)) return "If f(x)=x² then f'(x)=2x. At x=3, slope=6.";
    if (/probab|bayes|likeli/i.test(q)) return "If rain chance is 30% and forecast says rain, Bayes can update that chance.";
    if (/matrix|vector|linear/i.test(q)) return "Matrix [[1,2],[0,1]] scales x and shears along y; apply to (1,1).";
    if (/python|code|algorithm/i.test(q)) return "Write a function, add tests, and print results for quick feedback.";
    return "Write down knowns, draw a small diagram, and compute the missing quantity.";
  }

  function simulateType(text, cb) {
    const tokens = text.split("");
    let idx = 0;
    const chunk = Math.max(1, Math.floor(text.length / 120));
    const interval = setInterval(() => {
      idx += chunk;
      cb(tokens.slice(0, idx).join(""));
      if (idx >= tokens.length) clearInterval(interval);
    }, 12);
  }

  chatForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = String(chatInput?.value || "").trim();
    if (!text) return;
    appendMessage("user", text);
    if (chatInput) chatInput.value = "";

    // Simulate thinking
    const placeholder = document.createElement("div");
    placeholder.className = "msg msg--ai";
    placeholder.innerHTML = `<div>Thinking…</div><span class="msg__meta">AI Tutor</span>`;
    chatWindow?.appendChild(placeholder);
    chatWindow && (chatWindow.scrollTop = chatWindow.scrollHeight);

    await new Promise((r) => setTimeout(r, 350));
    const reply = think(text);

    // Type out reply
    simulateType(reply, (partial) => {
      placeholder.firstChild.textContent = partial;
      chatWindow && (chatWindow.scrollTop = chatWindow.scrollHeight);
    });
  });

  // Submit on Enter, newline with Shift+Enter
  chatInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm?.requestSubmit();
    }
  });
})();

