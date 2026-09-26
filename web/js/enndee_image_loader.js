/**
 * Load & Resize Image (Enndee) - resize-type aware widget visibility
 *
 * ComfyUI always renders every widget of a classic node. This extension shows
 * only the size options that matter for the selected resize type (mirroring the
 * DynamicCombo behaviour of ComfyUI core's Resize Image/Mask node) and hides
 * the resize settings entirely while "resize" is off.
 *
 * Hidden widgets keep their values, so saved workflows and the python node
 * still receive every parameter.
 */
import { app } from "/scripts/app.js";

// Widgets per resize type; must match the parameters used in
// nodes/enndee_image_loader.py::load and nodes/enndee_resize_modes.py.
const RESIZE_WIDGET_RULES = {
  "scale dimensions": ["width", "height", "keep_proportion", "divisible_by"],
  "scale by multiplier": ["multiplier", "divisible_by"],
  "scale longer dimension": ["longer_size", "divisible_by"],
  "scale shorter dimension": ["shorter_size", "divisible_by"],
  "scale width": ["width", "divisible_by"],
  "scale height": ["height", "divisible_by"],
  "scale total pixels": ["megapixels", "divisible_by"],
  "match size": ["keep_proportion", "divisible_by"],
  "scale to multiple": ["multiple"],
};

// Usable by every resize type.
const ALWAYS_VISIBLE_WHILE_RESIZING = ["resize_type", "scale_method", "no_upscale"];

// Widgets that switch the visible option set when their value changes.
const WATCHED_WIDGETS = new Set(["resize", "resize_type", "keep_proportion"]);

// Widgets this extension may collapse; everything else stays visible.
const HIDEABLE_WIDGETS = [
  "resize_type", "width", "height", "multiplier", "longer_size",
  "shorter_size", "megapixels", "multiple", "keep_proportion",
  "divisible_by", "background_color", "scale_method", "no_upscale",
];

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
      // Remember that *we* installed the overrides: most widgets do not define
      // computeSize/draw themselves, so the saved values can be undefined and a
      // restore test based on their value would never run (that bug left
      // previously hidden widgets collapsed after unhiding).
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

function applyResizeVisibility(node) {
  const find = (name) => node.widgets?.find((widget) => widget.name === name);
  const resizing = Boolean(find("resize")?.value);
  const resizeType = find("resize_type")?.value ?? "scale dimensions";
  // Unknown types fall back to the "scale dimensions" rules, like the python node.
  const rules = RESIZE_WIDGET_RULES[resizeType] ?? RESIZE_WIDGET_RULES["scale dimensions"];

  const wanted = new Set();
  if (resizing) {
    ALWAYS_VISIBLE_WHILE_RESIZING.forEach((name) => wanted.add(name));
    rules.forEach((name) => wanted.add(name));
    // background_color is only used when "keep_proportion" pads the image.
    if (rules.includes("keep_proportion") && find("keep_proportion")?.value) {
      wanted.add("background_color");
    }
    // Without a connected match input, "match size" falls back to width/height.
    const matchConnected = Boolean(node.inputs?.find((input) => input.name === "match")?.link);
    if (resizeType === "match size" && !matchConnected) {
      wanted.add("width");
      wanted.add("height");
    }
  }

  let changed = false;
  for (const name of HIDEABLE_WIDGETS) {
    const widget = find(name);
    if (!widget) continue;
    if (setWidgetHidden(widget, !wanted.has(name))) {
      changed = true;
    }
  }

  if (!changed) return;
  finishVisibilityUpdate(node);
}

app.registerExtension({
  name: "EnndeeImageLoader.ResizeVisibility",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "Enndee_ImageLoaderResize") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const refresh = () => applyResizeVisibility(node);
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
      for (const name of ["resize", "resize_type", "keep_proportion"]) {
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

      // Refresh when the "match" input is connected or removed.
      const onConnectionsChange = this.onConnectionsChange;
      this.onConnectionsChange = function () {
        const result = onConnectionsChange?.apply(this, arguments);
        setTimeout(refresh, 0);
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
