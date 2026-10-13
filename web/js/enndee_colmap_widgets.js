/**
 * COLMAP / GLOMAP Lichtfeld Tracker (Enndee) - dependent widget visibility.
 *
 * Widgets appear only while the widget they depend on is active:
 *   sharp_selection        -> Sharp Method / Sharp Batch / Sharp Num Frames
 *   use_rmbg               -> rmbg_mode / rmbg_threshold / rmbg_resize
 *   prepare_masks_for_face_detail -> face_mask_mode / face_prompt / face_threshold / face_mask_dilate
 *   mapper_backend=glomap  -> glomap_path (binary node only)
 *   image_format           -> jpeg_quality (JPEG) / embed_alpha_in_images (PNG)
 *   export_depth_maps      -> the dense-MVS block; fuse_dense_cloud / mesh_dense_surface
 *                            then gate their own options (both need the depth maps).
 *
 * `use_rmbg` is displayed as "Use RMBG for COLMAP" through widget.label (display only -
 * the serialized name stays `use_rmbg`, so saved workflows and scripts are untouched;
 * the same mechanism the Meridian parameters JS uses for the Spiral labels).
 *
 * Must stay in sync with glomap_lichtfeld_node.py / colmap_lichtfeld_node.py.
 */
import { app } from "/scripts/app.js";

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

// show: widgets that appear when EVERY condition in `when` holds. A widget listed in
// several groups is visible when ANY of its groups is satisfied.
const GROUPS = [
  { show: ["sharp_method", "sharp_batch", "sharp_num_frames"],
    when: [["sharp_selection", true]] },
  { show: ["rmbg_mode", "rmbg_threshold", "rmbg_resize"],
    when: [["use_rmbg", true]] },
  { show: ["face_mask_mode", "face_prompt", "face_threshold", "face_mask_dilate"],
    when: [["prepare_masks_for_face_detail", true]] },
  { show: ["glomap_path"], when: [["mapper_backend", "glomap"]] },
  { show: ["jpeg_quality"], when: [["image_format", "JPEG"]] },
  { show: ["embed_alpha_in_images"], when: [["image_format", "PNG"]] },
  { show: ["dense_max_image_size", "dense_geom_consistency", "fuse_dense_cloud",
           "mesh_dense_surface"],
    when: [["export_depth_maps", true]] },
  { show: ["dense_cloud_max_points"],
    when: [["fuse_dense_cloud", true], ["export_depth_maps", true]] },
  { show: ["mesh_method"],
    when: [["mesh_dense_surface", true], ["export_depth_maps", true]] },
];

// Display-only renames (widget.name stays as serialized).
const LABELS = [["use_rmbg", "Use RMBG for COLMAP"]];

const WATCHED_WIDGETS = new Set([
  "sharp_selection", "use_rmbg", "prepare_masks_for_face_detail",
  "mapper_backend", "image_format", "export_depth_maps",
  "fuse_dense_cloud", "mesh_dense_surface",
]);

function conditionsMet(node, when) {
  return when.every(([name, expected]) => {
    const widget = findWidget(node, name);
    return widget && widget.value === expected;
  });
}

function applyVisibility(node) {
  let changed = false;

  for (const [name, label] of LABELS) {
    const widget = findWidget(node, name);
    if (widget && widget.label !== label) {
      widget.label = label;
      changed = true;
    }
  }

  const visible = new Set();
  for (const group of GROUPS) {
    if (conditionsMet(node, group.when)) {
      group.show.forEach((name) => visible.add(name));
    }
  }
  const listed = new Set(GROUPS.flatMap((group) => group.show));
  for (const name of listed) {
    if (setWidgetHidden(findWidget(node, name), !visible.has(name))) {
      changed = true;
    }
  }

  if (changed) {
    finishVisibilityUpdate(node);
  }
}

app.registerExtension({
  name: "EnndeeColmapWidgets.DependentVisibility",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "Enndee_GLOMAPLichtfeldTracker" &&
        nodeData?.name !== "Enndee_ColmapLichtfeldTracker") return;

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