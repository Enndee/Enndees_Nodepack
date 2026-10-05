/**
 * COLMAP for Lichtfeld (Enndee) - live status label
 *
 * The node shows which backend it really uses (a CUDA pycolmap build or the CPU
 * fallback including the reason) and how far the reconstruction is. Both come from
 * the backend:
 *
 * - live while the node runs: the websocket event "enndee-colmap-status"
 *   (PromptServer.send_sync in nodes/colmap_lichtfeld_node.py),
 * - when it finishes: the built-in `{"ui": {"text": [...]}}` output, which ComfyUI
 *   renders as a text preview on the node.
 *
 * This extension makes sure the node has exactly one text widget and keeps it
 * updated live. Must stay in sync with nodes/colmap_lichtfeld_node.py.
 */
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_NAME = "Enndee_ColmapLichtfeldTracker";
const EVENT_NAME = "enndee-colmap-status";
// The built-in text preview of ComfyUI uses a widget called "text"; reusing that
// name means the final `ui.text` output lands in the same widget instead of a
// second one.
const WIDGET_NAME = "text";
const PLACEHOLDER = "pycolmap : checking the environment ...\nstatus   : idle";

function findStatusWidget(node) {
  return node.widgets?.find((widget) => widget.name === WIDGET_NAME);
}

function statusWidget(node) {
  let widget = findStatusWidget(node);
  if (widget) return widget;
  widget = node.addWidget("text", WIDGET_NAME, PLACEHOLDER, () => {}, {
    multiline: true,
  });
  if (!widget) return undefined;
  // Informational only: never store the status in the workflow file.
  widget.serialize = false;
  widget.serializeValue = async () => undefined;
  widget.disabled = true;
  widget.value = PLACEHOLDER;
  return widget;
}

function setStatus(node, text) {
  if (typeof text !== "string" || !text.length) return;
  const widget = statusWidget(node);
  if (!widget || widget.value === text) return;
  widget.value = text;
  try {
    // The textarea grows with the content (status + header lines).
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
  } catch (error) {
    // a size hiccup must never break the node
  }
  app.graph.setDirtyCanvas(true, true);
}

/** Keep a single status widget if the frontend added its own text preview. */
function dropDuplicateStatusWidgets(node) {
  const widgets = node.widgets ?? [];
  let seen = false;
  node.widgets = widgets.filter((widget) => {
    if (widget.name !== WIDGET_NAME) return true;
    if (seen) return false;
    seen = true;
    return true;
  });
}

app.registerExtension({
  name: "EnndeeColmapStatus.LiveLabel",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      statusWidget(this);
    };
  },

  setup() {
    api.addEventListener(EVENT_NAME, (event) => {
      const detail = event?.detail ?? {};
      const byId = detail.node ? app.graph.getNodeById(detail.node) : null;
      const node =
        byId ?? app.graph.nodes.find((candidate) => candidate.comfyClass === NODE_NAME);
      if (node) setStatus(node, detail.text);
    });

    // The built-in text preview may add a second widget - collapse it afterwards.
    api.addEventListener("executed", (event) => {
      const detail = event?.detail ?? {};
      const node = detail.node ? app.graph.getNodeById(detail.node) : null;
      if (node?.comfyClass === NODE_NAME) dropDuplicateStatusWidgets(node);
    });
  },
});
