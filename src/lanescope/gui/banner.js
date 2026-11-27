import { isAbort } from "./api.js";
import { $ } from "./dom.js";

export function showError(error) {
  if (isAbort(error)) {
    return;
  }
  const banner = $("banner");
  banner.textContent = error.message || String(error);
  banner.hidden = false;
}

export function hideError() {
  $("banner").hidden = true;
}
