import { $ } from "./dom.js";

const THEMES = ["system", "light", "dark"];

export function setupTheme() {
  let theme = "system";
  try {
    theme = localStorage.getItem("lanescope-theme") || "system";
  } catch {
    theme = "system";
  }
  applyTheme(THEMES.includes(theme) ? theme : "system");
  $("theme-toggle").addEventListener("click", () => {
    const current = document.documentElement.dataset.theme || "system";
    const next = THEMES[(THEMES.indexOf(current) + 1) % THEMES.length];
    applyTheme(next);
    try {
      localStorage.setItem("lanescope-theme", next);
    } catch {
      // The page works without a stored theme.
    }
  });
}

function applyTheme(theme) {
  if (theme === "system") {
    delete document.documentElement.dataset.theme;
  } else {
    document.documentElement.dataset.theme = theme;
  }
  $("theme-label").textContent = `Theme: ${theme}`;
}
