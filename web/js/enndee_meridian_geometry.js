/**
 * Meridian Geometry (Enndee) - dependent option visibility
 *
 * One node, two geometry backends. This extension shows only the controls that
 * matter for the selected `mode`, mirroring the Meridian Parameter Picker's
 * extension:
 *
 * - "VGGT preview (subprocess)": the VGGT controls (repo, python, cache,
 *   cache_dir plus the canvas / VGGT source-size / VGGT-path overrides) are
 *   visible and every fast-depth control is hidden - the fast widgets have no
 *   effect on the subprocess pass.
 * - "Fast depth (Depth-Anything-V2)": the VGGT controls disappear and the
 *   fast-depth group appears (model size, Depth-Anything-3 resolution cap,
 *   canvas, cloud/point density, the two cull rules).
 * - Master/detail rules mirror the picker's: the custom canvas numbers follow
 *   `canvas_mode` inside the fast group, `canvas_width`/`canvas_height` follow
 *   `canvas_enabled` and `full_size` follows `full_enabled` in the VGGT group.
 *
 * Hidden widgets keep their values, so saved workflows and the python node
 * still receive every parameter.
 */
import { app } from "/scripts/app.js";

// Must match VGGT_MODE / FAST_DEPTH_MODE in nodes/enndee_meridian_geometry.py.
const VGGT_MODE = "VGGT preview (subprocess)";
const FAST_DEPTH_MODE = "Fast depth (Depth-Anything-V2)";

// Controls that only matter to the in-process Depth-Anything backend (V2 or V3).
const FAST_DEPTH_PANEL = [
  "model_size", "canvas_mode", "custom_width", "custom_height", "cloud_scale",
  "point_size", "edge_cull", "edge_threshold", "back_face_cull", "depth_res",
];
// Subprocess-only controls: no effect while the fast backend renders. The canvas,
// source-size and VGGT-path overrides mirror the parameter picker's widgets; they only
// fill flags the args string did not set, so a connected picker keeps authority.
const VGGT_PANEL = [
  "repo", "python", "cache", "cache_dir",
  "canvas_enabled", "canvas_width", "canvas_height", "full_enabled", "full_size",
  "vggt_repo", "vggt_checkpoint",
];
// Master/detail: the explicit canvas numbers only while `canvas_mode` is "custom".
const CUSTOM_CANVAS_DETAILS = ["custom_width", "custom_height"];
// Master/detail in the VGGT group, the same pairs the picker hides behind its toggles.
const VGGT_CANVAS_DETAILS = ["canvas_width", "canvas_height"];
const VGGT_FULL_DETAILS = ["full_size"];

const ALL_HIDEABLE = [...new Set([...FAST_DEPTH_PANEL, ...VGGT_PANEL])];

const HIDDEN_WIDGET_SIZE = [0, -4];

// The classic canvas renderer hides widgets through "widget.hidden"; the Vue
// node renderer checks "widget.options.hidden" (its isWidgetVisible helper).
// Set both so classic and Vue node modes look the same, and keep the legacy
// computeSize/draw collapse for older frontends.
function setWidgetHidden(widget, hidden) {
  if ((widget.__enndeeHidden ?? false) === hidden) return false;
  widget.__enndeeHidden = hidden;
  widget.hidden = hidden;
  if (widget.options) {
    widget.options.hidden = hidden;
  }
  if (hidden) {
    if (!widget.__enndeeCollapsed) {
      widget.__enndeeCollapsed = true;
      widget.__enndeeComputeSize = widget.computeSize;
      widget.__enndeeDraw = widget.draw;
    }
    widget.computeSize = () => HIDDEN_WIDGET_SIZE;
    widget.draw = () => {};
  } else if (widget.__enndeeCollapsed) {
    widget.__enndeeCollapsed = false;
    if (widget.__enndeeComputeSize === undefined) {
      delete widget.computeSize;
    } else {
      widget.computeSize = widget.__enndeeComputeSize;
    }
    if (widget.__enndeeDraw === undefined) {
      delete widget.draw;
    } else {
      widget.draw = widget.__enndeeDraw;
    }
    delete widget.__enndeeComputeSize;
    delete widget.__enndeeDraw;
  }
  return true;
}

function finishVisibilityUpdate(node) {
  const computed = node.computeSize?.();
  if (computed && node.size) {
    node.setSize?.([Math.max(node.size[0], computed[0]), computed[1]]);
  }
  for (const widget of node.widgets ?? []) {
    widget.triggerDraw?.();
  }
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
  app.canvas?.setDirty?.(true, true);
  // Bump the graph version like the frontend does after widget changes so the
  // node stores re-evaluate widget visibility.
  node.graph?.incrementVersion?.();
}

function applyGeometryVisibility(node) {
  const find = (name) => node.widgets?.find((widget) => widget.name === name);
  const mode = find("mode")?.value ?? VGGT_MODE;
  const fast = mode === FAST_DEPTH_MODE;
  const visible = new Set(ALL_HIDEABLE);

  if (fast) {
    VGGT_PANEL.forEach((name) => visible.delete(name));
  } else {
    FAST_DEPTH_PANEL.forEach((name) => visible.delete(name));
    if (!Boolean(find("canvas_enabled")?.value)) {
      VGGT_CANVAS_DETAILS.forEach((name) => visible.delete(name));
    }
    if (!Boolean(find("full_enabled")?.value)) {
      VGGT_FULL_DETAILS.forEach((name) => visible.delete(name));
    }
  }
  if (!fast || (find("canvas_mode")?.value ?? "auto_meridian480") !== "custom") {
    CUSTOM_CANVAS_DETAILS.forEach((name) => visible.delete(name));
  }

  let changed = false;
  for (const name of ALL_HIDEABLE) {
    const widget = find(name);
    if (!widget) continue;
    if (setWidgetHidden(widget, !visible.has(name))) {
      changed = true;
    }
  }

  if (!changed) return;
  finishVisibilityUpdate(node);
}

// Widgets that change the visible set when their value changes.
const WATCHED_WIDGETS = new Set(["mode", "canvas_mode", "canvas_enabled", "full_enabled"]);

app.registerExtension({
  name: "EnndeeMeridianGeometry.DependentVisibility",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "Enndee_MeridianGeometry") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const refresh = () => applyGeometryVisibility(node);
      // Immediate pass plus deferred passes: classic and Vue node rendering
      // commit widget values in different orders.
      const scheduleRefresh = () => {
        refresh();
        setTimeout(refresh, 0);
        if (typeof requestAnimationFrame === "function") {
          requestAnimationFrame(refresh);
        }
      };

      // Classic widget callbacks (used by canvas widgets and by the Vue
      // update handler, which forwards to the live widget callback).
      for (const name of WATCHED_WIDGETS) {
        const widget = this.widgets?.find((candidate) => candidate.name === name);
        if (!widget) continue;
        const original = widget.callback;
        widget.callback = function (value) {
          const result = original?.apply(this, arguments);
          scheduleRefresh();
          return result;
        };
      }

      // The frontend also reports widget changes through this node hook.
      const onWidgetChanged = this.onWidgetChanged;
      this.onWidgetChanged = function (name) {
        const result = onWidgetChanged?.apply(this, arguments);
        if (WATCHED_WIDGETS.has(name)) {
          scheduleRefresh();
        }
        return result;
      };

      // Refresh after loading a saved workflow or pasting the node.
      const onConfigure = this.onConfigure;
      this.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        setTimeout(refresh, 0);
        return result;
      };

      // Safety net: keep the visible options in sync even if a frontend path
      // does not report a change. The diff is cheap and only touches the
      // widgets when something actually moved.
      const timer = setInterval(refresh, 400);
      const clear = () => clearInterval(timer);
      this.addEventListener?.("removed", clear);
      const onRemoved = this.onRemoved;
      this.onRemoved = function () {
        const result = onRemoved?.apply(this, arguments);
        clear();
        return result;
      };

      setTimeout(refresh, 0);
    };
  },
});
