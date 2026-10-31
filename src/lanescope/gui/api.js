export class RequestError extends Error {}

// The demo on GitHub Pages runs LaneScope inside the page with Pyodide in place of the
// server, as a meta tag of that page says.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
let browserApi = null;

export async function getJSON(path, params = {}, signal = undefined) {
  if (inBrowser()) {
    return answer("GET", path, params, signal);
  }
  const query = new URLSearchParams(params).toString();
  const response = await fetch(query ? `${path}?${query}` : path, { signal });
  return read(response);
}

export async function postJSON(path, body, signal = undefined) {
  if (inBrowser()) {
    return answer("POST", path, body, signal);
  }
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return read(response);
}

function inBrowser() {
  return document.querySelector('meta[name="lanescope-backend"]')?.content === "pyodide";
}

async function startBrowserApi() {
  const { loadPyodide } = await import(`${PYODIDE}pyodide.mjs`);
  const pyodide = await loadPyodide({ indexURL: PYODIDE });
  await pyodide.loadPackage(["numpy", "micropip"]);
  const wheel = document.querySelector('meta[name="lanescope-wheel"]').content;
  pyodide.globals.set("lanescope_wheel", new URL(wheel, document.baseURI).href);
  await pyodide.runPythonAsync("import micropip\nawait micropip.install(lanescope_wheel, deps=False)");
  const api = pyodide.pyimport("lanescope.gui.browser").BrowserApi("/demo");
  document.getElementById("pyodide-status")?.remove();
  return api;
}

async function answer(method, path, payload, signal) {
  browserApi ??= startBrowserApi();
  const api = await browserApi;
  if (signal?.aborted) {
    throw new DOMException("A newer request replaced this one", "AbortError");
  }
  const { status, data, error } = JSON.parse(api.request(method, path, JSON.stringify(payload)));
  if (status !== 200) {
    throw new RequestError(error || `the page answered ${status}`);
  }
  return data;
}

async function read(response) {
  let data;
  try {
    data = await response.json();
  } catch (error) {
    // An abort can land while the body arrives, and it stays an abort.
    if (isAbort(error)) {
      throw error;
    }
    throw new RequestError(`the server answered ${response.status} without JSON`);
  }
  if (!response.ok) {
    throw new RequestError(data.error || `the server answered ${response.status}`);
  }
  return data;
}

// Runs one request of a kind at a time and cancels the older one, so a fast
// slider never shows the answer of an earlier position.
export class Latest {
  constructor() {
    this.controller = null;
  }

  abort() {
    this.controller?.abort();
    this.controller = null;
  }

  async run(request) {
    this.controller?.abort();
    const controller = new AbortController();
    this.controller = controller;
    try {
      return await request(controller.signal);
    } finally {
      if (this.controller === controller) {
        this.controller = null;
      }
    }
  }
}

export function isAbort(error) {
  return error instanceof DOMException && error.name === "AbortError";
}
