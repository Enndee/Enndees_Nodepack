/**
 * Meridian Parameters and Camera (Enndee) - dependent option visibility
 *
 * The node builds the Geometry arguments plus a camera path in one of two modes:
 *
 * - "Manual path": the path group is shown. Inside it, "path_camera_mode" decides the
 *   style panel - "O Orbits" (station list + orbit diameter), "Alternating Height"
 *   (yaw pair, arc elevations, switches) or "Spiral Sweep" (yaw pair, sweep elevations).
 *   The style switcher, the look pivot and the dolly are shared.
 * - "Automatic (estimated)": the path group disappears and the automatic controls
 *   (target scene/subject plus the speed cap) appear; the estimator uses the
 *   reference_image / subject_mask inputs if they are connected.
 *
 * Hidden widgets keep their values, so switching modes never loses settings.
 * Must stay in sync with enndee_meridian_parameters.py.
 */
import { app } from "/scripts/app.js";

const O_ORBIT_PANEL = ["path_orbit_front", "path_orbit_left", "path_orbit_right", "path_orbit_back", "path_orbit_up", "path_orbit_down", "path_orbit_left_back", "path_orbit_right_back", "path_start_station", "path_orbit_diameter"];
const HEIGHT_SWEEP_PANEL = ["path_start_yaw", "path_target_yaw", "path_low_elevation", "path_high_elevation", "path_arc_switches", "path_first_arc"];
const SPIRAL_SWEEP_PANEL = ["path_start_yaw", "path_target_yaw", "path_spiral_start_elevation", "path_spiral_end_elevation"];
const PATH_SHARED = ["path_camera_mode", "path_dolly", "path_pivot_x", "path_pivot_y", "path_pivot_z"];
const PATH_PANEL = [...O_ORBIT_PANEL, ...HEIGHT_SWEEP_PANEL, ...SPIRAL_SWEEP_PANEL, ...PATH_SHARED];
const AUTO_PANEL = ["auto_target", "auto_max_speed"];
const ALL_HIDEABLE = [...PATH_PANEL, ...AUTO_PANEL];

// Must match CAMERA_MODE_OPTIONS in nodes/enndee_meridian_camera_path.py ...
const O_ORBIT_MODE = "O Orbits";
const HEIGHT_SWEEP_MODE = "Alternating Height";
const SPIRAL_SWEEP_MODE = "Spiral Sweep";
// ... and CAMERA_MODES in nodes/enndee_meridian_parameters.py.
const MANUAL_MODE = "Manual path";
const AUTOMATIC_MODE = "Automatic (estimated)";
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

function applyVisibility(node) {
  const visible = new Set(["output_frames", "camera_mode", "auto_target", "auto_max_speed", "cull"]);
  if (String(findWidget(node, "camera_mode")?.value) !== AUTOMATIC_MODE) {
    AUTO_PANEL.forEach((name) => visible.delete(name));
    PATH_SHARED.forEach((name) => visible.add(name));
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

  if (!changed) return;
  finishVisibilityUpdate(node);
}

// Widgets that change the visible set when their value changes.
const WATCHED_WIDGETS = new Set(["camera_mode", "path_camera_mode"]);

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
