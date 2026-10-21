export class RequestError extends Error {}

export async function getJSON(path, params = {}, signal = undefined) {
  const query = new URLSearchParams(params).toString();
  const response = await fetch(query ? `${path}?${query}` : path, { signal });
  return read(response);
}

export async function postJSON(path, body, signal = undefined) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return read(response);
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
