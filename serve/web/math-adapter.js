// Select one locally bundled viewer. Both share the same source-preserving core.
"use strict";
const unavailableMathAdapter = {
  async initialize() { throw new Error("No math engine installed"); },
  async render() { return {ok: false, reason: "unavailable"}; },
};
globalThis.StrataMathEngines = {
  names: Object.freeze(["katex", "mathjax"]),
  select(name) {
    if (!this.names.includes(name)) name = "katex";
    globalThis.StrataMathAdapter = globalThis.StrataMathAdapters?.[name] || unavailableMathAdapter;
    StrataMath.reset();
    return name;
  },
};
StrataMathEngines.select("katex");
