import { state } from "./state.js";

export function readHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  const row = params.get("row");
  return {
    trace: params.get("trace"),
    pc: params.get("pc"),
    space: params.get("space"),
    row: row === null ? null : Number(row),
  };
}

export function writeHash() {
  const params = new URLSearchParams({
    trace: state.traceId,
    pc: state.selected.pc,
    space: state.selected.space,
    row: String(state.row),
  });
  history.replaceState(null, "", `#${params}`);
}

export function writeTraceHash(id, { push = false } = {}) {
  const hash = `#${new URLSearchParams({ trace: id })}`;
  if (push) {
    history.pushState(null, "", hash);
  } else {
    history.replaceState(null, "", hash);
  }
}

export function clearHash({ push = false } = {}) {
  if (location.hash) {
    const bare = location.pathname + location.search;
    if (push) {
      history.pushState(null, "", bare);
    } else {
      history.replaceState(null, "", bare);
    }
  }
}
