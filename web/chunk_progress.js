import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "Viggle.ChunkProgress",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["ViggleSampleChunk", "ViggleChunkedSampler"].includes(nodeData.name)) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            const widget = ComfyWidgets.STRING(this, "live_progress",
                ["STRING", { multiline: true }], app).widget;
            widget.inputEl.readOnly = true;
            widget.options.serialize = false;
            widget.value = "Waiting for chunk sampling";
            this.viggleProgressWidget = widget;
            this.viggleHasCheckpoints = nodeData.name === "ViggleSampleChunk";
            return result;
        };
    },
    setup() {
        const active = new Map();
        api.addEventListener("viggle.chunk_progress", ({ detail }) => {
            const node = app.graph.getNodeById(detail.node_id);
            if (!node?.viggleProgressWidget) return;
            node.viggleProgressWidget.value = detail.text;
            active.set(node, detail.prompt_id);
            app.graph.setDirtyCanvas(true, true);
        });
        for (const event of ["execution_error", "execution_interrupted"]) {
            api.addEventListener(event, ({ detail }) => {
                for (const [node, promptId] of active) {
                    if (promptId !== detail.prompt_id) continue;
                    node.viggleProgressWidget.value += node.viggleHasCheckpoints
                        ? "\nRun stopped; saved checkpoints remain on disk."
                        : "\nRun stopped; chunk reuse is memory-only.";
                    active.delete(node);
                }
                app.graph.setDirtyCanvas(true, true);
            });
        }
        api.addEventListener("execution_start", () => active.clear());
        api.addEventListener("execution_success", () => active.clear());
    },
});
