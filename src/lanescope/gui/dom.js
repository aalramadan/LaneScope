export const $ = (id) => document.getElementById(id);

export function stat(label, value, title = "") {
  const wrapper = document.createElement("div");
  wrapper.title = title;
  const term = document.createElement("dt");
  term.textContent = label;
  const detail = document.createElement("dd");
  detail.textContent = value;
  wrapper.append(term, detail);
  return wrapper;
}

export function cell(content, className) {
  const element = document.createElement("td");
  element.className = className;
  element.textContent = content;
  return element;
}

// A bar under a count, as long as the count against the largest one in its table.
export function meter(value, peak) {
  const bar = document.createElement("span");
  bar.className = "meter";
  const fill = document.createElement("span");
  fill.style.width = `${(100 * value) / Math.max(peak, 1)}%`;
  bar.append(fill);
  return bar;
}

// A second, smaller line of a table cell, such as the ideal count under a count.
export function detail(text, className = "where") {
  const element = document.createElement("span");
  element.className = className;
  element.textContent = text;
  return element;
}
