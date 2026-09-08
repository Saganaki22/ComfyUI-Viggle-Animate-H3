import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const handlers = new Map();
const nodes = new Map();
let extension;
let redraws = 0;
const app = {
    registerExtension(value) { extension = value; },
    graph: { getNodeById: id => nodes.get(String(id)), setDirtyCanvas: () => redraws++ },
};
const api = { addEventListener: (event, callback) => handlers.set(event, callback) };
const ComfyWidgets = { STRING: () => ({ widget: { inputEl: {}, options: {} } }) };
const source = readFileSync(new URL("../web/chunk_progress.js", import.meta.url), "utf8");
vm.runInNewContext(source.replace(/^import .*;\r?\n/gm, ""), { app, api, ComfyWidgets });
class Node { onNodeCreated() { this.originalCalled = true; } }
await extension.beforeRegisterNodeDef(Node, { name: "ViggleSampleChunk" });
extension.setup();
const node = new Node();
node.onNodeCreated();
nodes.set("9", node);
assert.equal(node.originalCalled, true);
assert.equal(node.viggleProgressWidget.inputEl.readOnly, true);
assert.equal(node.viggleProgressWidget.options.serialize, false);
const emit = (event, detail) => handlers.get(event)({ detail });
for (let chunk = 1; chunk <= 4; chunk++) {
    emit("viggle.chunk_progress", { node_id: "9", prompt_id: "run", text: `Chunk ${chunk} of 4` });
    assert.equal(node.viggleProgressWidget.value, `Chunk ${chunk} of 4`);
}
emit("execution_error", { prompt_id: "unrelated" });
assert.equal(node.viggleProgressWidget.value, "Chunk 4 of 4");
emit("execution_interrupted", { prompt_id: "run" });
assert.match(node.viggleProgressWidget.value, /Run stopped/);
emit("viggle.chunk_progress", { node_id: "missing", prompt_id: "run", text: "ignored" });
emit("viggle.chunk_progress", { node_id: "9", prompt_id: "next", text: "restoring" });
emit("execution_success", { prompt_id: "next" });
emit("execution_error", { prompt_id: "next" });
assert.equal(node.viggleProgressWidget.value, "restoring");
assert.ok(redraws >= 4);
class ChunkedNode {}
await extension.beforeRegisterNodeDef(ChunkedNode, { name: "ViggleChunkedSampler" });
const chunked = new ChunkedNode();
chunked.onNodeCreated();
nodes.set("20", chunked);
emit("viggle.chunk_progress", { node_id: "20", prompt_id: "chunked", text: "Decoding final video" });
assert.equal(chunked.viggleProgressWidget.value, "Decoding final video");
emit("execution_interrupted", { prompt_id: "chunked" });
assert.match(chunked.viggleProgressWidget.value, /memory-only/);
assert.doesNotMatch(chunked.viggleProgressWidget.value, /on disk/);
console.log("Frontend progress checks passed: updates, read-only/nonserialized widget, routing, interruption, lifecycle.");
