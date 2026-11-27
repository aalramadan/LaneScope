import { $ } from "./dom.js";
import * as format from "./format.js";
import { sameBuffer, state } from "./state.js";

// The fix panel draws on the diagnosis and its button acts on the fix panel, so the page
// hands over these actions at start.
let actions = { apply: null, hide: null };

export function setFixActions({ apply, hide }) {
  actions = { apply, hide };
}

export function renderDiagnosis() {
  const holder = $("diagnosis");
  const { selected, access, overview } = state;
  holder.hidden = !(selected && access && selected.space === access.space);
  if (holder.hidden) {
    return;
  }
  const verdict = format.diagnosis(selected, access, overview.totals);
  holder.className = `diagnosis ${verdict.tone}`;
  const head = document.createElement("div");
  head.className = "diagnosis-head";
  const icon = document.createElement("span");
  icon.className = "diagnosis-icon";
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = verdict.tone === "warn" ? "⚠︎" : "✓";
  head.append(icon);
  const title = document.createElement("strong");
  title.className = "diagnosis-title";
  title.textContent = verdict.title;
  head.append(title);
  if (verdict.badge) {
    const badge = document.createElement("span");
    badge.className = "diagnosis-badge";
    badge.textContent = verdict.badge;
    head.append(badge);
  }
  const cause = document.createElement("p");
  cause.textContent = verdict.cause;
  holder.replaceChildren(head, cause);
  if (verdict.cost) {
    const cost = document.createElement("p");
    cost.className = "diagnosis-cost";
    cost.textContent = verdict.cost;
    holder.append(cost);
  }
  if (verdict.tone === "warn") {
    holder.append(fixLine(verdict.advice));
  }
}

function fixLine(advice) {
  const line = document.createElement("div");
  line.className = "fix-line";
  const what = document.createElement("p");
  line.append(what);
  if (advice) {
    what.textContent = advice;
    return line;
  }
  const { suggestions } = state;
  if (state.selected?.data_dependent) {
    what.textContent = format.DATA_DEPENDENT;
    const [item] = suggestions && sameBuffer(suggestions.buffer, state.buffer) ? suggestions.changes : [];
    if (item) {
      const removed = format.count(item.excessive_before - item.excessive_after);
      what.textContent +=
        ` On this recording, the best change, ${format.fixName(item.params)}, removes ${removed} ` +
        `of the ${format.count(item.excessive_before)} extra passes on this buffer.`;
    }
    return line;
  }
  if (!suggestions || !sameBuffer(suggestions.buffer, state.buffer)) {
    what.textContent = state.suggestionError || "Looking for a padding or a swizzle that removes the conflicts...";
    return line;
  }
  const [item] = suggestions.changes;
  if (!item) {
    what.textContent = "No padding or swizzle of this buffer lowers its extra passes.";
    return line;
  }
  what.textContent = `Fix by ${format.fixName(item.params)}. This ${format.fixEffect(item)}.`;
  const shown = Object.entries(item.params).every(([key, value]) => state.change[key] === value);
  const button = document.createElement("button");
  button.type = "button";
  button.className = "primary-button";
  button.textContent = shown ? "Hide the fix" : "Show this fix";
  button.setAttribute("aria-pressed", String(shown));
  button.addEventListener("click", () => (shown ? actions.hide() : actions.apply(item)));
  line.append(button);
  return line;
}
