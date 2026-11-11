// Student Support Assistant - chat page logic (Ahmed)
// Plain JavaScript: no framework, no build step.
// Talks to POST /ask using the agreed contract (docs/ask-api-examples.json).
// Privacy: the chat only exists in this page's memory. Nothing is put in
// localStorage except the light/dark choice (see theme.js).

"use strict";

// ---------- Settings ----------
const ASK_URL = "/ask";        // starts with "/", so it works behind Nginx over HTTPS
const MAX_CHARS = 500;         // same limit as the /ask contract
const TIMEOUT_MS = 60000;      // answers can take 15-30+ s on CPU, so wait up to 60 s
const STILL_WORKING_MS = 15000;

// Topics for the sidebar (desktop) and the grid (mobile). Each sends an example question.
const TOPICS = [
  { label: "Accommodation",          question: "How do I apply for university accommodation?" },
  { label: "Fees and funding",       question: "How do I pay my tuition fees?" },
  { label: "Library",                question: "When is the library open at weekends?" },
  { label: "IT help",                question: "How do I reset my university password?" },
  { label: "Wellbeing",              question: "What wellbeing support does the university offer?" },
  { label: "Enrolment",              question: "How do I enrol at the start of the year?" },
  { label: "Campus facilities",      question: "Where can I print on campus?" },
  { label: "Food and places nearby", question: "Where can I get lunch near City Campus?" },
];

// Contact rows on the wellbeing card. Written by people, never by the AI.
// [CHECK] Copied on 11 Nov 2026 from:
//   shu.ac.uk/myhallam/help-in-an-emergency           (999, Security)
//   shu.ac.uk/myhallam/university-life/staying-safe/contact-security  (24 hours)
//   shu.ac.uk/wellbeing/urgent-help                   (Samaritans, Shout)
//   shu.ac.uk/myhallam/wellbeing                      (Student Wellbeing hours)
// Re-check every line on the live pages before the demo.
const CRISIS_CONTACTS = [
  { title: "Call 999", tel: "999",
    detail: "In an immediate or life-threatening emergency" },
  { title: "Campus Security: 0114 225 2000", tel: "01142252000",
    detail: "24 hours, City Campus and Collegiate Crescent. 888 from an internal phone." },
  { title: "Samaritans: 116 123", tel: "116123",
    detail: "A 24-hour support service to discuss personal issues" },
  { title: "Shout: text STEEL to 85258", sms: "85258",
    detail: "A 24/7 text service if you're experiencing a personal crisis or are unable to cope" },
  { title: "Student Wellbeing: 0114 225 2136", tel: "01142252136",
    detail: "Mon, Tue, Thu, Fri 9am to 5pm. Wed 11am to 5pm. Lines close 12:30 to 1:30pm." },
];

// Friendly names for OpenStreetMap place types
const PLACE_TYPES = {
  cafe: "Café",
  restaurant: "Restaurant",
  fast_food: "Fast food",
  supermarket: "Supermarket",
  convenience: "Convenience shop",
};

// ---------- Page elements ----------
const messagesEl = document.getElementById("messages");
const form = document.getElementById("composer");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send");
const counter = document.getElementById("counter");
const hint = document.getElementById("input-hint");
const welcomeTemplate = document.getElementById("welcome-template");
const themeBtn = document.getElementById("theme-toggle");

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let busy = false;   // true while waiting for (or typing out) an answer

// ---------- Helpers ----------
// Builds an element and sets its text with textContent, never innerHTML,
// so nothing from the server or the user can inject HTML or scripts.
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Only allow normal web links (blocks "javascript:" and other odd URLs)
function safeUrl(url) {
  try {
    const u = new URL(url);
    return u.protocol === "https:" || u.protocol === "http:" ? u.href : null;
  } catch {
    return null;
  }
}

// Link that opens in a new tab and tells screen-reader users so
function externalLink(url, className) {
  const a = el("a", className);
  a.href = url;
  a.target = "_blank";
  a.rel = "noopener";            // the new tab can't control this page
  return a;
}
function newTabNote() {
  return el("span", "visually-hidden", " (opens in a new tab)");
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ---------- Adding messages ----------
function addMessage(who, bubbleClass) {
  const row = el("div", "msg " + who);
  const bubble = el("div", "bubble" + (bubbleClass ? " " + bubbleClass : ""));
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollToBottom();
  return { row, bubble };
}

function addUserMessage(text) {
  const { bubble } = addMessage("user");
  bubble.appendChild(el("p", "text", text));
}

function addErrorMessage(text) {
  const { bubble } = addMessage("bot", "error");
  bubble.appendChild(el("p", "text", text));
}

// Typing animation: shows the answer word by word.
// Screen readers get the full text at once from a hidden copy, and the
// animated copy is hidden from them, so it isn't read out word by word.
async function typeInto(paragraph, text) {
  if (reduceMotion) {
    paragraph.textContent = text;
    return;
  }
  paragraph.appendChild(el("span", "visually-hidden", text));
  const shown = el("span", "typing");
  shown.setAttribute("aria-hidden", "true");
  paragraph.appendChild(shown);

  const words = text.split(/(\s+)/);                 // keep the spaces
  const delay = Math.max(8, Math.min(35, 1500 / words.length)); // whole answer in ~1.5 s or less
  for (const word of words) {
    shown.textContent += word;
    scrollToBottom();
    await wait(delay);
  }
  shown.classList.remove("typing");
}

function formatSeconds(ms) {
  if (typeof ms !== "number") return null;
  return "Answered in " + (ms / 1000).toFixed(1) + " s";
}

// Normal answer: typed text, then sources / places / time fade in
async function renderAnswer(data, question) {
  const { bubble } = addMessage("bot");
  const p = el("p", "text");
  bubble.appendChild(p);
  await typeInto(p, data.answer);

  const extras = el("div", "extras");
  if (Array.isArray(data.sources) && data.sources.length > 0) {
    extras.appendChild(renderSources(data.sources));
  }
  if (Array.isArray(data.places) && data.places.length > 0) {
    extras.appendChild(renderPlaces(data.places));
    extras.appendChild(renderPlaceFollowUps(question));
  }
  const time = formatSeconds(data.time_ms);
  if (time) extras.appendChild(el("p", "meta", time));
  bubble.appendChild(extras);
  scrollToBottom();
}

function renderSources(sources) {
  const wrap = el("div", "sources");
  wrap.appendChild(el("p", "sources-label", "Sources"));
  const list = el("ul", "source-list");
  for (const s of sources) {
    const url = safeUrl(s.url);
    if (!url) continue;
    const li = el("li");
    const a = externalLink(url, "source-chip");
    a.appendChild(el("span", "source-title", s.title || url));
    a.appendChild(el("span", "source-domain", new URL(url).hostname.replace(/^www\./, "")));
    a.appendChild(newTabNote());
    li.appendChild(a);
    list.appendChild(li);
  }
  wrap.appendChild(list);
  return wrap;
}

function renderPlaces(places) {
  const frag = document.createDocumentFragment();
  const list = el("ul", "places");
  for (const p of places) {
    const card = el("li", "place-card");
    const info = el("div");
    info.appendChild(el("p", "place-name", p.name));
    const parts = [];
    if (p.type) parts.push(PLACE_TYPES[p.type] || p.type.replace(/_/g, " "));
    if (typeof p.distance_m === "number") parts.push(Math.round(p.distance_m) + " m");
    info.appendChild(el("p", "place-info", parts.join(" · ")));
    card.appendChild(info);
    const url = safeUrl(p.map_url);
    if (url) {
      const a = externalLink(url);
      a.textContent = "Map";
      a.appendChild(el("span", "visually-hidden", " for " + p.name + " (opens in a new tab)"));
      card.appendChild(a);
    }
    list.appendChild(card);
  }
  frag.appendChild(list);
  // OpenStreetMap's licence requires this credit
  frag.appendChild(el("p", "osm-note", "Distances are straight-line. Map data © OpenStreetMap contributors."));
  return frag;
}

// Follow-up chips just send a new question, so no contract change is needed.
// places.py works out the campus and place type from the words.
function renderPlaceFollowUps(question) {
  const q = (question || "").toLowerCase();
  const campus = q.includes("collegiate") ? "Collegiate Crescent" : "City Campus";
  const other = campus === "City Campus" ? "Collegiate Crescent" : "City Campus";
  const askedCafes = /caf[eé]|coffee/.test(q);

  const chips = el("div", "chips");
  chips.appendChild(questionChip("Near " + other + " instead",
    (askedCafes ? "Cafés near " : "Places to eat near ") + other));
  chips.appendChild(askedCafes
    ? questionChip("All places to eat", "Places to eat near " + campus)
    : questionChip("Just cafés", "Cafés near " + campus));
  return chips;
}

function questionChip(label, question) {
  const b = el("button", "chip", label);
  b.type = "button";
  b.dataset.question = question;
  return b;
}

// Wellbeing response: appears at once (no typing animation), in a calm,
// distinct card, with fixed human-written text and fixed contacts.
function renderSafety(data) {
  const { bubble } = addMessage("bot", "safety");
  bubble.setAttribute("role", "region");
  bubble.setAttribute("aria-label", "Wellbeing support");
  bubble.appendChild(el("h3", null, "You don't have to deal with this alone"));
  bubble.appendChild(el("p", "text", data.answer));

  const contacts = el("ul", "contacts");
  for (const c of CRISIS_CONTACTS) {
    const li = el("li");
    const title = el("strong");
    if (c.tel || c.sms) {
      const a = el("a", null, c.title);
      a.href = c.tel ? "tel:" + c.tel : "sms:" + c.sms;
      title.appendChild(a);
    } else {
      title.textContent = c.title;
    }
    li.appendChild(title);
    li.appendChild(el("span", "contact-detail", c.detail));
    contacts.appendChild(li);
  }
  bubble.appendChild(contacts);

  if (Array.isArray(data.sources) && data.sources.length > 0) {
    bubble.appendChild(renderSources(data.sources));
  }

  const notes = el("div", "safety-notes");
  notes.appendChild(el("p", null, "This is a set message written by people, not the AI."));
  notes.appendChild(el("p", null, "Nothing you type here is saved."));
  bubble.appendChild(notes);
}

// Loading bubble: bouncing dots and a message that changes after 15 s
function addLoading() {
  const loading = addMessage("bot", "loading");
  const dots = el("span", "dots");
  dots.setAttribute("aria-hidden", "true");
  dots.append(el("span"), el("span"), el("span"));
  const text = el("span", "text", "Searching university pages. This can take up to 15 seconds.");
  loading.bubble.append(dots, text);
  const timer = setTimeout(() => {
    text.textContent = "Still working on it. Some answers take up to 30 seconds.";
  }, STILL_WORKING_MS);
  return { remove() { clearTimeout(timer); loading.row.remove(); } };
}

// ---------- Talking to /ask ----------
async function ask(question) {
  if (busy) return;

  question = question.trim();
  if (!question) {
    showHint("Please type a question first.");
    input.focus();
    return;
  }
  if (question.length > MAX_CHARS) {
    showHint("Please keep your question to " + MAX_CHARS + " characters or fewer.");
    return;
  }

  showHint("");
  removeWelcome();
  addUserMessage(question);
  input.value = "";
  updateCounter();
  setBusy(true);

  const loading = addLoading();
  const controller = new AbortController();            // lets us give up after TIMEOUT_MS
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const res = await fetch(ASK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });
    clearTimeout(timer);
    loading.remove();

    if (res.status === 429) {
      // Sent by Nginx, may be HTML rather than JSON, so don't read it
      addErrorMessage("Too many questions at once. Please wait a few seconds and try again.");
      return;
    }

    const data = await readJson(res);

    if (res.ok && data && typeof data.answer === "string") {
      if (data.safety === true) {
        renderSafety(data);
      } else {
        await renderAnswer(data, question);
      }
    } else if (res.status === 400) {
      addErrorMessage((data && data.error) || "Please enter a question (max 500 characters).");
    } else {
      // 500, 502/503/504 from Nginx if the app is down, or anything unexpected
      addErrorMessage((data && data.error) || "Something went wrong. Please try again.");
    }
  } catch (err) {
    loading.remove();
    if (err.name === "AbortError") {
      addErrorMessage("That took too long to answer. Please try again, or try a shorter question.");
    } else {
      addErrorMessage("Couldn't reach the assistant. Check your connection and try again.");
    }
  } finally {
    clearTimeout(timer);
    setBusy(false);
    input.focus();
  }
}

// Returns parsed JSON, or null if the body isn't JSON (e.g. an Nginx HTML error page)
async function readJson(res) {
  try {
    return await res.json();
  } catch {
    return null;
  }
}

// ---------- UI state ----------
function setBusy(value) {
  busy = value;
  sendBtn.disabled = value;
  document.querySelectorAll("button[data-question]").forEach((b) => (b.disabled = value));
}

function showHint(text) {
  hint.textContent = text;
}

function updateCounter() {
  const n = input.value.length;
  counter.textContent = n + " / " + MAX_CHARS;
  counter.classList.toggle("over", n >= MAX_CHARS);
}

function removeWelcome() {
  const w = document.getElementById("welcome");
  if (w) w.remove();
}

function showWelcome() {
  messagesEl.replaceChildren(welcomeTemplate.content.cloneNode(true));
  const grid = messagesEl.querySelector("[data-topic-grid]");
  for (const t of TOPICS) grid.appendChild(questionButton(t));
}

function buildSidebar() {
  const list = document.getElementById("topic-list");
  for (const t of TOPICS) {
    const li = el("li");
    li.appendChild(questionButton(t));
    list.appendChild(li);
  }
}

function questionButton(topic) {
  const b = el("button", null, topic.label);
  b.type = "button";
  b.dataset.question = topic.question;
  return b;
}

// "New chat" clears everything (it was only ever in memory)
function newChat() {
  if (busy) return;
  input.value = "";
  updateCounter();
  showHint("");
  showWelcome();
  input.focus();
}

// ---------- Light / dark mode ----------
function currentTheme() {
  const chosen = document.documentElement.getAttribute("data-theme");
  if (chosen) return chosen;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function showThemeState() {
  themeBtn.setAttribute("aria-pressed", String(currentTheme() === "dark"));
}

function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("theme", next); } catch { /* storage blocked: fine */ }
  showThemeState();
}

// ---------- Events ----------
form.addEventListener("submit", (e) => {
  e.preventDefault();
  ask(input.value);
});

// Enter sends, Shift+Enter makes a new line
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    ask(input.value);
  }
});

input.addEventListener("input", () => {
  updateCounter();
  if (hint.textContent) showHint("");
});

// One listener for every button that carries a question
// (sidebar topics, mobile grid, "Try asking" chips, places follow-ups)
document.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-question]");
  if (b && !b.disabled) ask(b.dataset.question);
});

// "How it works" and "Privacy" panels
document.querySelectorAll("[data-open]").forEach((b) => {
  b.addEventListener("click", () => document.getElementById(b.dataset.open).showModal());
});

document.getElementById("new-chat").addEventListener("click", newChat);
themeBtn.addEventListener("click", toggleTheme);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", showThemeState);

// ---------- Start ----------
buildSidebar();
showWelcome();
updateCounter();
showThemeState();
