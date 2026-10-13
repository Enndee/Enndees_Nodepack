/**
 * Lichtfeld Headless Trainer (Enndee) - dependent widget visibility.
 *
 * The node carries ~60 widgets and most of them only mean anything once a switch is
 * set: the depth-loss parameters are dead weight until "Use Depth Loss" is on, the
 * normal-loss ones until "Use Normal Loss" is on, the anchor ones until "Use Surface
 * Anchors" is on, and so on. This extension hides the dependent rows while their switch
 * is off, so the panel shows only the controls the current choices actually use.
 *
 * Hidden widgets KEEP their values - switching a group off and back on never loses a
 * setting, and a saved workflow keeps every value it was saved with. Nothing here
 * changes what the node reads: the backend already ignores a parameter whose switch is
 * off (e.g. `--depth-loss-weight` only goes out with `--use-depth-loss`), so hiding is
 * purely a display change.
 *
 * Must stay in sync with nodes/lichtfeld_training_node.py (see HIDEABLE_WIDGETS there);
 * the test `test_javascript_mirrors_the_hideable_widgets` reads both and fails if they
 * drift.
 */
import { app } from "/scripts/app.js";

const NODE_NAME = "Enndee_LichtfeldHeadlessTrainer";

/**
 * The visibility rules. Each entry names the widgets that are only useful while `when`
 * holds; `value(name)` reads any widget's current value (undefined when the widget is
 * absent). A widget may appear in more than one rule - it is then hidden unless every
 * rule that owns it is satisfied, which is how `freeze_lr_scale` needs both the anchors
 * and their freeze on.
 */
const RULES = [
  {
    // The depth term only exists once depth supervision is on.
    when: (value) => value("use_depth_loss") === true,
    widgets: ["depth_loss_mode", "depth_loss_weight"],
  },
  {
    // The normal term only exists once normal supervision is on.
    when: (value) => value("use_normal_loss") === true,
    widgets: ["normal_loss_weight", "normal_consistency_weight",
              "normal_flatten_weight", "normal_loss_space"],
  },
  {
    // Anchors: the mesh, its budget and the freeze all describe the anchor splats.
    when: (value) => value("use_surface_anchors") === true,
    widgets: ["surface_anchor_mesh", "anchor_resolution", "anchor_freeze"],
  },
  {
    // The partial-unfreeze dial only exists while the anchors are actually frozen.
    when: (value) => value("use_surface_anchors") === true && value("anchor_freeze") === true,
    widgets: ["freeze_lr_scale"],
  },
  {
    // Evaluation: the steps and the comparison images are output of a run that
    // evaluates. `eval_steps` is itself one of the switches (supplying steps turns
    // evaluation on), so it keeps its own rule visible as soon as it has a value.
    when: (value) => value("enable_eval") === true
      || String(value("evaluation") ?? "off") !== "off"
      || String(value("eval_steps") ?? "").trim() !== "",
    widgets: ["eval_steps", "save_eval_images"],
  },
  {
    // Downscaling: "Disable Downscaling" forces resize factor 1 and no width cap, so
    // the two knobs it overrides are noise.
    when: (value) => value("disable_downscaling") !== true,
    widgets: ["image_resize_factor", "max_image_width"],
  },
  {
    // Only a solid background has a colour to set.
    when: (value) => String(value("background_mode") ?? "solidcolor") === "solidcolor",
    widgets: ["background_color"],
  },
  {
    // The opacity penalty pushes down the masked-out region; with the masks ignored
    // there is no masked-out region.
    when: (value) => String(value("mask_mode") ?? "none") !== "none",
    widgets: ["mask_opacity_penalty"],
  },
  {
    // Inverting is meaningless when the masks are not read at all - and Studio reads
    // `segment_and_ignore` as bands, where the invert is ignored too.
    when: (value) => {
      const mode = String(value("mask_mode") ?? "none");
      return mode !== "none" && mode !== "segment_and_ignore";
    },
    widgets: ["invert_masks"],
  },
];

function findWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

/**
 * Collapse a widget to zero height (hidden) or restore it. Returns whether the widget
 * actually changed, so a refresh can skip the expensive resize when nothing moved.
 * @param {object} widget LiteGraph widget.
 * @param {boolean} hidden Whether the widget should be invisible.
 * @returns {boolean} True when the state changed.
 */
function setWidgetHidden(widget, hidden) {
  if (!widget) return false;
  const collapsed = widget.__enndeeCollapsed === true;
  if (hidden === collapsed) return false;
  if (hidden) {
    widget.__enndeeOriginalComputeSize = widget.computeSize;
    widget.computeSize = () => [0, -4];
    widget.__enndeeOriginalDraw = widget.draw;
    widget.draw = () => {};
    widget.hidden = true;
    widget.__enndeeCollapsed = true;
  } else {
    if (widget.__enndeeOriginalComputeSize === undefined) {
      delete widget.computeSize;
    } else {
      widget.computeSize = widget.__enndeeOriginalComputeSize;
    }
    if (widget.__enndeeOriginalDraw === undefined) {
      delete widget.draw;
    } else {
      widget.draw = widget.__enndeeOriginalDraw;
    }
    delete widget.__enndeeOriginalComputeSize;
    delete widget.__enndeeOriginalDraw;
    widget.hidden = false;
    widget.__enndeeCollapsed = false;
  }
  return true;
}

function finishVisibilityUpdate(node) {
  const computed = node.computeSize?.();
  if (computed) {
    node.setSize?.([node.size?.[0] ?? computed[0], computed[1]]);
  }
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

/**
 * Show the widgets whose rules hold and hide the rest. Widgets named by no rule are
 * always shown; a widget named by several rules is shown only when all of them hold.
 * @param {object} node The trainer node.
 * @returns {boolean} True when anything changed.
 */
function applyVisibility(node) {
  const value = (name) => findWidget(node, name)?.value;
  const hidden = new Set();
  for (const rule of RULES) {
    let satisfied = false;
    try {
      satisfied = rule.when(value) === true;
    } catch (_error) {
      satisfied = true;                 // a half-built node: show, never hide by accident
    }
    if (!satisfied) {
      rule.widgets.forEach((name) => hidden.add(name));
    }
  }
  let changed = false;
  for (const rule of RULES) {
    for (const name of rule.widgets) {
      if (setWidgetHidden(findWidget(node, name), hidden.has(name))) {
        changed = true;
      }
    }
  }
  if (changed) {
    finishVisibilityUpdate(node);
  }
  return changed;
}

app.registerExtension({
  name: "EnndeeLichtfeldTrainer.DependentVisibility",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const refresh = () => applyVisibility(node);
      // Immediate pass plus deferred passes: classic and Vue node rendering commit
      // widget values in different orders.
      const scheduleRefresh = () => {
        refresh();
        setTimeout(refresh, 0);
        if (typeof requestAnimationFrame === "function") {
          requestAnimationFrame(refresh);
        }
      };

      const onWidgetChanged = this.onWidgetChanged;
      this.onWidgetChanged = function (name) {
        const result = onWidgetChanged?.apply(this, arguments);
        if (RULES.some((rule) => rule.widgets.includes(name))) {
          scheduleRefresh();
        }
        return result;
      };

      const onConfigure = this.onConfigure;
      this.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        setTimeout(refresh, 0);
        return result;
      };

      // Safety net for frontend paths that do not report a change; the diff is cheap
      // and only touches widgets that actually moved.
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
