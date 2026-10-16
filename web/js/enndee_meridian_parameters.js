/** Meridian Parameters and Camera (Enndee) dependent option visibility. */
import { app } from "/scripts/app.js";

const O_ORBIT_PANEL = ["path_orbit_front", "path_orbit_left", "path_orbit_right",
  "path_orbit_back", "path_orbit_up", "path_orbit_down", "path_orbit_left_back",
  "path_orbit_right_back", "path_start_station", "path_orbit_diameter"];
const HEIGHT_SWEEP_PANEL = ["path_start_yaw", "path_target_yaw", "path_low_elevation",
  "path_high_elevation", "path_arc_switches", "path_first_arc"];
const SPIRAL_SWEEP_PANEL = ["path_start_yaw", "path_target_yaw",
  "path_spiral_start_elevation", "path_spiral_end_elevation"];
const PATH_SHARED = ["path_camera_mode", "path_dolly", "path_pivot_x", "path_pivot_y",
  "path_pivot_z"];
const PATH_PIVOTS = ["path_pivot_x", "path_pivot_y", "path_pivot_z"];
const PATH_PANEL = [...O_ORBIT_PANEL, ...HEIGHT_SWEEP_PANEL, ...SPIRAL_SWEEP_PANEL,
  ...PATH_SHARED];
const AUTO_PANEL = ["auto_target", "auto_max_speed", "auto_path_mode", "auto_subject_fill",
  "auto_scene_survey_width", "auto_orbit_view_angle", "auto_orbit_coverage",
  "auto_orbit_direction", "auto_pivot_x",
  "auto_pivot_y", "auto_pivot_z", "auto_orbit_angle", "spiral_end", "spiral_slope",
  "band_start_angle", "band_count", "band_width", "band_start_yaw", "band_stop_angle",
  "auto_orbit_look"];
const FACE_PANEL = ["face_fill", "face_yaw_span", "face_prompt", "face_threshold",
  "face_orbit_mode"];
const ALL_HIDEABLE = [...PATH_PANEL, ...AUTO_PANEL, ...FACE_PANEL];
const FACE_SWITCH = "face_orbit";
const MODE_PANELS = new Map([
  ["O Orbits", O_ORBIT_PANEL],
  ["Alternating Height", HEIGHT_SWEEP_PANEL],
  ["Spiral Sweep", SPIRAL_SWEEP_PANEL],
]);
const O_ORBIT_COVERAGES = new Set(["O-Orbit Front", "O-Orbit Front&Back"]);
const BAND_COVERAGES = new Set(["Banded Orbits", "Banded Orbits Front & Back"]);
const WIDGET_LABELS = [
  ["auto_orbit_angle", "O-Orbit Angle"],
  ["spiral_end", "Spiral Winding"],
  ["spiral_slope", "Spiral Center Slope"],
  ["band_start_angle", "Band Start Angle"],
  ["band_count", "Band Count"],
  ["band_width", "Band Width (Yaw)"],
  ["band_start_yaw", "Band Start Yaw"],
  ["band_stop_angle", "Band Stop Angle"],
];

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
    if (widget.__enndeeOriginalComputeSize === undefined) delete widget.computeSize;
    else widget.computeSize = widget.__enndeeOriginalComputeSize;
    if (widget.__enndeeOriginalDraw === undefined) delete widget.draw;
    else widget.draw = widget.__enndeeOriginalDraw;
    delete widget.__enndeeOriginalComputeSize;
    delete widget.__enndeeOriginalDraw;
    widget.hidden = false;
    widget.__enndeeCollapsed = false;
  }
  return true;
}

function finishVisibilityUpdate(node) {
  const computed = node.computeSize?.();
  if (computed) node.setSize?.([node.size?.[0] ?? computed[0], computed[1]]);
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

function applyCoverageUI(node, coverage) {
  let changed = false;
  const banded = BAND_COVERAGES.has(coverage);
  if (!banded) {
    for (const name of ["band_start_angle", "band_count", "band_width", "band_start_yaw",
      "band_stop_angle"]) {
      if (setWidgetHidden(findWidget(node, name), true)) changed = true;
    }
  }
  for (const [name, label] of WIDGET_LABELS) {
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
  const automatic = String(findWidget(node, "camera_mode")?.value) === "Automatic";
  const pathMode = String(findWidget(node, "auto_path_mode")?.value ?? "Automatic");
  const target = String(findWidget(node, "auto_target")?.value ?? "subject");
  let changed = false;
  let coverage = String(findWidget(node, "auto_orbit_coverage")?.value ?? "");
  if (automatic) {
    AUTO_PANEL.forEach((name) => visible.add(name));
    if (target !== "scene" || pathMode !== "Automatic") {
      visible.delete("auto_scene_survey_width");
    }
    if (!O_ORBIT_COVERAGES.has(coverage)) {
      ["auto_orbit_view_angle", "auto_orbit_direction", "auto_orbit_angle"]
        .forEach((name) => visible.delete(name));
    }
    if (coverage !== "Spiral") {
      ["spiral_end", "spiral_slope"].forEach((name) => visible.delete(name));
    }
    if (String(findWidget(node, FACE_SWITCH)?.value) === "Auto (SAM3)") {
      FACE_PANEL.forEach((name) => visible.add(name));
    }
    if (BAND_COVERAGES.has(coverage)) {
      ["band_start_angle", "band_count", "band_width", "band_start_yaw", "band_stop_angle"]
        .forEach((name) => visible.add(name));
    } else {
      ["band_start_angle", "band_count", "band_width", "band_start_yaw", "band_stop_angle"]
        .forEach((name) => visible.delete(name));
    }
  }
  if (!automatic || pathMode === "Manual") {
    const shared = automatic
      ? PATH_SHARED.filter((name) => !PATH_PIVOTS.includes(name))
      : PATH_SHARED;
    shared.forEach((name) => visible.add(name));
    const style = String(findWidget(node, "path_camera_mode")?.value);
    (MODE_PANELS.get(style) ?? O_ORBIT_PANEL).forEach((name) => visible.add(name));
  }
  for (const name of ALL_HIDEABLE) {
    const widget = findWidget(node, name);
    if (widget && setWidgetHidden(widget, !visible.has(name))) changed = true;
  }
  if (automatic && applyCoverageUI(node, coverage)) changed = true;
  if (changed) finishVisibilityUpdate(node);
}

const WATCHED_WIDGETS = new Set(["camera_mode", "path_camera_mode", "auto_path_mode", "auto_target",
  "auto_orbit_coverage", FACE_SWITCH]);

app.registerExtension({
  name: "EnndeeMeridianParameters.DependentVisibility",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === "Enndee_MeridianGeometry") return;
    if (nodeData?.name !== "Enndee_MeridianParametersAndCamera") return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const refresh = () => applyVisibility(node);
      const scheduleRefresh = () => {
        refresh();
        setTimeout(refresh, 0);
        if (typeof requestAnimationFrame === "function") requestAnimationFrame(refresh);
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
        if (WATCHED_WIDGETS.has(name)) scheduleRefresh();
        return result;
      };
      const onConfigure = this.onConfigure;
      this.onConfigure = function () {
        const result = onConfigure?.apply(this, arguments);
        setTimeout(refresh, 0);
        return result;
      };
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