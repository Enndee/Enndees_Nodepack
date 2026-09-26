/**
 * Meridian Parameter Picker (Enndee) - dependent option visibility
 *
 * The picker builds both the Geometry arguments and (optionally) a custom
 * O-orbit camera path. This extension shows only the controls that matter:
 *
 * - "Use Custom Camera" off: the camera-path group is hidden and every args
 *   control is shown (with its own master/detail rules).
 * - "Use Custom Camera" on: the camera-path group appears while the freeze,
 *   source-start, follow and authored camera-move / pivot options disappear,
 *   because Meridian Geometry overrides them whenever a custom path is
 *   connected.
 * - The camera-path group follows "path_camera_mode": "O Orbits" shows the
 *   station list plus the orbit diameter, "Alternating Height" shows the yaw
 *   pair, the two arc elevations and the switch controls, "Spiral Sweep" shows
 *   the yaw pair and the start/end elevation; the mode switcher and the look
 *   pivot are always visible while the group is open.
 * - Detail groups follow their master toggle: the freeze window only while
 *   "Freeze Source" is on ("Freeze Length" only while "Freeze Full Output" is
 *   off), pivot and pivot-lock only while "Enable Pivot" is on, the secondary
 *   pivot only while "Aim Camera" is on, smoothing only while "Follow" is on,
 *   and the canvas / VGGT source sizes only while their overrides are enabled.
 *
 * Hidden widgets keep their values, so saved workflows and the python node
 * still receive every parameter.
 */
import { app } from "/scripts/app.js";

// Camera-path controls; shown only while "use_custom_camera" is on. The style
// panels are exclusive: "O Orbits" shows the station list plus the orbit
// diameter, "Alternating Height" the pendulum arcs, "Spiral Sweep" the monotone
// sweep elevations; the mode switcher and the look pivot are shared. Must match
// the "path_*" widget names in enndee_meridian_parameter_picker.py.
const O_ORBIT_PANEL = ["path_orbit_front", "path_orbit_left", "path_orbit_right", "path_orbit_back", "path_orbit_up", "path_orbit_down", "path_orbit_left_back", "path_orbit_right_back", "path_start_station", "path_orbit_diameter"];
const HEIGHT_SWEEP_PANEL = ["path_start_yaw", "path_target_yaw", "path_low_elevation", "path_high_elevation", "path_arc_switches", "path_first_arc"];
const SPIRAL_SWEEP_PANEL = ["path_start_yaw", "path_target_yaw", "path_spiral_start_elevation", "path_spiral_end_elevation"];
const CUSTOM_CAMERA_SHARED = ["path_camera_mode", "path_dolly", "path_pivot_x", "path_pivot_y", "path_pivot_z"];
const CUSTOM_CAMERA_PANEL = [...O_ORBIT_PANEL, ...HEIGHT_SWEEP_PANEL, ...SPIRAL_SWEEP_PANEL, ...CUSTOM_CAMERA_SHARED];
// Must match CAMERA_MODE_OPTIONS in nodes/enndee_meridian_camera_path.py.
const O_ORBIT_MODE = "O Orbits";
const HEIGHT_SWEEP_MODE = "Alternating Height";
const SPIRAL_SWEEP_MODE = "Spiral Sweep";
const MODE_PANELS = new Map([
  [O_ORBIT_MODE, O_ORBIT_PANEL],
  [HEIGHT_SWEEP_MODE, HEIGHT_SWEEP_PANEL],
  [SPIRAL_SWEEP_MODE, SPIRAL_SWEEP_PANEL],
]);

// Ignored by Meridian Geometry while a custom camera path is connected.
const HIDDEN_IN_CUSTOM_MODE = ["source_start", "freeze_source", "freeze_frame", "freeze_full_output", "freeze_length", "yaw_from", "yaw_to", "truck", "boom", "dolly", "zoom", "pivot_enabled", "pivot_x", "pivot_y", "pivot_lock", "aim", "pivot_to_enabled", "pivot_to_x", "pivot_to_y", "path_mode", "ease", "live_speed", "fast_back", "follow", "smooth", "camera_path"];

// Detail groups (args mode only); their master toggle must be on to matter.
const FREEZE_DETAILS = ["freeze_frame", "freeze_full_output"];
const PIVOT_DETAILS = ["pivot_x", "pivot_y", "pivot_lock"];
const PIVOT_TO_DETAILS = ["pivot_to_x", "pivot_to_y"];
const FOLLOW_DETAILS = ["smooth"];
const CANVAS_DETAILS = ["canvas_width", "canvas_height"];
const FULL_DETAILS = ["full_size"];

const DETAIL_GROUPS = [...FREEZE_DETAILS, ...PIVOT_DETAILS, ...PIVOT_TO_DETAILS, ...FOLLOW_DETAILS, ...CANVAS_DETAILS, ...FULL_DETAILS];

const ALL_HIDEABLE = [...new Set([...CUSTOM_CAMERA_PANEL, ...HIDDEN_IN_CUSTOM_MODE, ...DETAIL_GROUPS])];

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

function applyPickerVisibility(node) {
  const find = (name) => node.widgets?.find((widget) => widget.name === name);
  const custom = Boolean(find("use_custom_camera")?.value);
  const visible = new Set(ALL_HIDEABLE);

  if (custom) {
    HIDDEN_IN_CUSTOM_MODE.forEach((name) => visible.delete(name));
    // The style panels are exclusive, but they may share widgets (both sweep
    // panels contain the yaw pair), so clear every panel first and re-add only
    // the active one - deleting just the inactive lists would hide the shared
    // widgets as well.
    const activePanel = MODE_PANELS.get(find("path_camera_mode")?.value) ?? MODE_PANELS.get(O_ORBIT_MODE);
    for (const panel of MODE_PANELS.values()) {
      panel.forEach((name) => visible.delete(name));
    }
    activePanel.forEach((name) => visible.add(name));
  } else {
    CUSTOM_CAMERA_PANEL.forEach((name) => visible.delete(name));
  }

  // Master/detail rules; they are no-ops while custom-camera mode hides the group.
  const freezeOn = !custom && Boolean(find("freeze_source")?.value);
  if (!freezeOn) {
    FREEZE_DETAILS.forEach((name) => visible.delete(name));
  }
  if (!freezeOn || Boolean(find("freeze_full_output")?.value)) {
    visible.delete("freeze_length");
  }
  if (!Boolean(find("pivot_enabled")?.value)) {
    PIVOT_DETAILS.forEach((name) => visible.delete(name));
  }
  const aimOn = Boolean(find("aim")?.value);
  if (!aimOn) {
    visible.delete("pivot_to_enabled");
  }
  if (!aimOn || !Boolean(find("pivot_to_enabled")?.value)) {
    PIVOT_TO_DETAILS.forEach((name) => visible.delete(name));
  }
  if (!Boolean(find("follow")?.value)) {
    FOLLOW_DETAILS.forEach((name) => visible.delete(name));
  }
  if (!Boolean(find("canvas_enabled")?.value)) {
    CANVAS_DETAILS.forEach((name) => visible.delete(name));
  }
  if (!Boolean(find("full_enabled")?.value)) {
    FULL_DETAILS.forEach((name) => visible.delete(name));
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
const WATCHED_WIDGETS = new Set([
  "use_custom_camera", "path_camera_mode", "freeze_source", "freeze_full_output",
  "pivot_enabled", "aim", "pivot_to_enabled", "follow", "canvas_enabled",
  "full_enabled",
]);

app.registerExtension({
  name: "EnndeeMeridianPicker.DependentVisibility",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "Enndee_MeridianParameterPicker") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const refresh = () => applyPickerVisibility(node);
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

