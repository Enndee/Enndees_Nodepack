/**
 * Meridian Parameters and Camera (Enndee) - dependent option visibility
 *
 * The node builds the Geometry arguments plus a camera path in one of two modes:
 *
 * - "Manual": the path group is shown. Inside it, "path_camera_mode" decides the style panel -
 *   "O Orbits" (station list + orbit diameter), "Alternating Height" (yaw pair, arc elevations,
 *   switches) or "Spiral Sweep" (yaw pair, sweep elevations). The style switcher, the look pivot
 *   and the dolly are shared.
 * - "Automatic": the automatic controls appear (target scene/subject, the speed cap, the path
 *   mode, Auto Subject Fill, the orbit's view angle/coverage/direction and the pivot offsets).
 *   "auto_path_mode" then decides the path:
 *   "Automatic" hides the manual group (the estimator builds the path itself), "Manual" shows it -
 *   the same styles, but flown around the estimated pivot, so the absolute look-pivot widgets stay
 *   hidden.
 *
 * Hidden widgets keep their values, so switching modes never loses settings.
 * Must stay in sync with enndee_meridian_parameters.py.
 */
import { app } from "/scripts/app.js";

const O_ORBIT_PANEL = ["path_orbit_front", "path_orbit_left", "path_orbit_right", "path_orbit_back", "path_orbit_up", "path_orbit_down", "path_orbit_left_back", "path_orbit_right_back", "path_start_station", "path_orbit_diameter"];
const HEIGHT_SWEEP_PANEL = ["path_start_yaw", "path_target_yaw", "path_low_elevation", "path_high_elevation", "path_arc_switches", "path_first_arc"];
const SPIRAL_SWEEP_PANEL = ["path_start_yaw", "path_target_yaw", "path_spiral_start_elevation", "path_spiral_end_elevation"];
const PATH_SHARED = ["path_camera_mode", "path_dolly", "path_pivot_x", "path_pivot_y", "path_pivot_z"];
const PATH_PIVOTS = ["path_pivot_x", "path_pivot_y", "path_pivot_z"];
const PATH_PANEL = [...O_ORBIT_PANEL, ...HEIGHT_SWEEP_PANEL, ...SPIRAL_SWEEP_PANEL, ...PATH_SHARED];
const AUTO_PANEL = ["auto_target", "auto_max_speed", "auto_path_mode", "auto_subject_fill", "auto_orbit_view_angle", "auto_orbit_coverage", "auto_orbit_direction", "auto_pivot_x", "auto_pivot_y", "auto_pivot_z", "auto_orbit_angle", "spiral_end", "spiral_slope"];
// Deprecated widgets: never visible (the saved value survives in the graph, the backend ignores it).
// `auto_orbit_distance` was the fixed camera stand-off; the distance follows Auto Subject Fill now.
// `auto_orbit_end` was where the concluding orbit stopped and `auto_orbit_size` scaled the O - the
// fixed front circle plus Auto Orbit View Angle / Auto Orbit Coverage say where the path goes now.
const DEPRECATED_PANEL = ["auto_orbit_distance", "auto_orbit_end", "auto_orbit_size"];
const ALL_HIDEABLE = [...PATH_PANEL, ...AUTO_PANEL, ...DEPRECATED_PANEL];

// Must match CAMERA_MODE_OPTIONS in nodes/enndee_meridian_camera_path.py ...
const O_ORBIT_MODE = "O Orbits";
const HEIGHT_SWEEP_MODE = "Alternating Height";
const SPIRAL_SWEEP_MODE = "Spiral Sweep";
// ... and CAMERA_MODES / AUTO_PATH_MODES in nodes/enndee_meridian_parameters.py.
const MANUAL_MODE = "Manual";
const AUTOMATIC_MODE = "Automatic";
const AUTOMATIC_PATH = "Automatic";
const MANUAL_PATH = "Manual";
const MODE_PANELS = new Map([
  [O_ORBIT_MODE, O_ORBIT_PANEL],
  [HEIGHT_SWEEP_MODE, HEIGHT_SWEEP_PANEL],
  [SPIRAL_SWEEP_MODE, SPIRAL_SWEEP_PANEL],
]);

function findWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

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

// The Auto Orbit Angle slot doubles as the Spiral End Angle inside the Spiral coverage: the label
// and the slider range follow the coverage, so the widget always reads as what it does right now
// (the value itself is kept - the backend clamps each meaning to its own window; the panel only
// nudges a value that is still the other mode's *default*). The ranges must stay in sync with
// FRONT_ORBIT_ANGLE_MIN / FRONT_ORBIT_LIMIT and SPIRAL_END_ARC_MIN / _MAX in
// enndee_meridian_auto_camera.py, the labels with the widget tooltips in
// enndee_meridian_parameters.py.
const SPIRAL_COVERAGE = "Spiral";
const SHARED_ANGLE_WIDGET = "auto_orbit_angle";
const ORBIT_ANGLE_LABEL = "O Orbit Angle";
const ORBIT_ANGLE_RANGE = [5, 60];
const ORBIT_ANGLE_DEFAULT = 45;
const SPIRAL_END_ANGLE_LABEL = "Spiral End Angle";
const SPIRAL_END_ANGLE_RANGE = [5, 90];
const SPIRAL_END_ANGLE_DEFAULT = 90;
const SPIRAL_WINDING_LABEL = "Spiral Winding";
const SPIRAL_SLOPE_LABEL = "Spiral Center Slope";

function setSpiralLabels(node, spiral) {
  let changed = false;
  const shared = findWidget(node, SHARED_ANGLE_WIDGET);
  if (shared) {
    const label = spiral ? SPIRAL_END_ANGLE_LABEL : ORBIT_ANGLE_LABEL;
    const [low, high] = spiral ? SPIRAL_END_ANGLE_RANGE : ORBIT_ANGLE_RANGE;
    if (shared.label !== label) {
      shared.label = label;
      changed = true;
    }
    if (shared.options && (shared.options.min !== low || shared.options.max !== high)) {
      shared.options.min = low;
      shared.options.max = high;
      changed = true;
    }
    // A value still sitting on the *other* mode's default is not a user choice: carry it to this
    // mode's default so picking Spiral opens on the full 90 deg spiral, and leaving it restores 45.
    const fallback = spiral ? SPIRAL_END_ANGLE_DEFAULT : ORBIT_ANGLE_DEFAULT;
    const stranded = spiral ? ORBIT_ANGLE_DEFAULT : SPIRAL_END_ANGLE_DEFAULT;
    if (Number(shared.value) === stranded && Number(shared.value) !== fallback) {
      shared.value = fallback;
      changed = true;
    }
  }
  for (const [name, label] of [["spiral_end", SPIRAL_WINDING_LABEL],
                               ["spiral_slope", SPIRAL_SLOPE_LABEL]]) {
    const widget = findWidget(node, name);
    if (widget && widget.label !== label) {
      widget.label = label;
      changed = true;
    }
  }
  return changed;
}

function applyVisibility(node) {
  const visible = new Set(["output_frames", "camera_mode"]);
  const cameraMode = String(findWidget(node, "camera_mode")?.value);
  const automatic = cameraMode === AUTOMATIC_MODE;
  const pathMode = String(findWidget(node, "auto_path_mode")?.value ?? AUTOMATIC_PATH);
  if (automatic) {
    AUTO_PANEL.forEach((name) => visible.add(name));
  }
  if (!automatic || pathMode === MANUAL_PATH) {
    // Automatic camera mode can still fly the manual path - then only the pivot widgets stay
    // hidden, because the estimate already placed the look-pivot they would otherwise set.
    const shared = automatic
      ? PATH_SHARED.filter((name) => !PATH_PIVOTS.includes(name))
      : PATH_SHARED;
    shared.forEach((name) => visible.add(name));
    const style = String(findWidget(node, "path_camera_mode")?.value);
    (MODE_PANELS.get(style) ?? O_ORBIT_PANEL).forEach((name) => visible.add(name));
  }

  let changed = false;
  for (const name of ALL_HIDEABLE) {
    const widget = findWidget(node, name);
    if (!widget) continue;
    if (setWidgetHidden(widget, !visible.has(name))) {
      changed = true;
    }
  }
  // The shared angle widget reads as the Spiral End Angle while the Spiral coverage is picked (and
  // the spiral's own widgets get their labels).
  if (automatic && setSpiralLabels(
        node, String(findWidget(node, "auto_orbit_coverage")?.value) === SPIRAL_COVERAGE)) {
    changed = true;
  }

  if (!changed) return;
  finishVisibilityUpdate(node);
}

// Widgets that change the visible set (or the shared widget's meaning) when their value changes.
const WATCHED_WIDGETS = new Set(["camera_mode", "path_camera_mode", "auto_path_mode",
                                 "auto_orbit_coverage"]);

app.registerExtension({
  name: "EnndeeMeridianParameters.DependentVisibility",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "Enndee_MeridianParametersAndCamera") return;

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

      const onWidgetChanged = this.onWidgetChanged;
      this.onWidgetChanged = function (name) {
        const result = onWidgetChanged?.apply(this, arguments);
        if (WATCHED_WIDGETS.has(name)) {
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
