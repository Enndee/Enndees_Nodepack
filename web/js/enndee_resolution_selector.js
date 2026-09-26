/**
 * Resolution Selector (Enndee) - live preview widget
 *
 * Renders the same "2048 × 2048   4.00 MP" preview the built-in ComfyUI
 * Resolution Selector shows. The calculation mirrors the python node, so the
 * preview updates live while dragging the megapixels slider (no queue run
 * needed). Resize types other than "scale dimensions" are reported with their
 * own widget value, because the connected image size is unknown in the browser.
 *
 * When "keep_source_aspect_ratio" is enabled and an image is connected,
 * the source dimensions are not available in the browser, so show the MP target.
 */
import { app } from "/scripts/app.js";

// label -> [widthRatio, heightRatio]  (must match nodes/enndee_resolution_selector.py)
const ASPECT_RATIOS = {
  "1:1 (Square)": [1, 1],
  "2:3 (Portrait Photo)": [2, 3],
  "3:2 (Photo)": [3, 2],
  "3:4 (Portrait Standard)": [3, 4],
  "4:3 (Standard)": [4, 3],
  "9:16 (Portrait Widescreen)": [9, 16],
  "16:9 (Widescreen)": [16, 9],
  "21:9 (Ultrawide)": [21, 9],
};

const MEGAPIXEL_BASE = 1024 * 1024;

function snapToMultiple(value, multiple) {
  const step = Math.max(1, Math.round(multiple) || 1);
  return Math.max(step, Math.round(value / step) * step);
}

function dimensionsFromRatio(widthRatio, heightRatio, megapixels, multiple) {
  const totalPixels = megapixels * MEGAPIXEL_BASE;
  const scale = Math.sqrt(totalPixels / (widthRatio * heightRatio));
  return [
    snapToMultiple(widthRatio * scale, multiple),
    snapToMultiple(heightRatio * scale, multiple),
  ];
}

// Human-readable hints for the resize types of ComfyUI's Resize Image/Mask node.
const RESIZE_TYPE_LABELS = {
  "scale by multiplier": (v) => `× ${v.toFixed(2)} of the source size`,
  "scale longer dimension": (v) => `longer edge ${Math.round(v)} px`,
  "scale shorter dimension": (v) => `shorter edge ${Math.round(v)} px`,
  "scale width": () => "selected width, source aspect",
  "scale height": () => "selected height, source aspect",
  "scale total pixels": (v) => `${v.toFixed(2)} MP on the source aspect`,
  "match size": () => "reference 'match' input size",
  "scale to multiple": (v) => `multiple of ${Math.round(v)} px`,
};

function computePreview(ratioLabel, megapixels, multiple, keepSource, hasSourceImage, resizeType, values) {
  if (resizeType && resizeType !== "scale dimensions") {
    const describe = RESIZE_TYPE_LABELS[resizeType];
    const value =
      resizeType === "scale by multiplier" ? values.multiplier
      : resizeType === "scale longer dimension" ? values.longerSize
      : resizeType === "scale shorter dimension" ? values.shorterSize
      : resizeType === "scale total pixels" ? values.megapixels
      : resizeType === "scale to multiple" ? values.multiple
      : null;
    const detail = describe ? describe(Number(value) || 0) : "connected image";
    return { text: resizeType, detail, dim: true };
  }
  if (keepSource && hasSourceImage) {
    return {
      text: "source aspect ratio",
      detail: `${megapixels.toFixed(2)} MP target`,
      dim: true,
    };
  }
  const ratio = ASPECT_RATIOS[ratioLabel] || ASPECT_RATIOS["1:1 (Square)"];
  const [w, h] = dimensionsFromRatio(ratio[0], ratio[1], megapixels, multiple);
  return {
    text: `${w} × ${h}`,
    detail: `${((w * h) / MEGAPIXEL_BASE).toFixed(2)} MP`,
    dim: false,
  };
}

function buildPreviewElement() {
  const wrapper = document.createElement("div");
  wrapper.style.cssText = [
    "box-sizing:border-box",
    "width:100%",
    "padding:8px 6px",
    "margin:4px 0",
    "border-radius:6px",
    "background:rgba(0,0,0,0.25)",
    "text-align:center",
    "font-family:system-ui,sans-serif",
    "font-size:14px",
    "line-height:1.25",
    "user-select:none",
    "pointer-events:none",
  ].join(";");

  const size = document.createElement("span");
  size.style.cssText = "font-weight:600;color:#e5e7eb;";
  const pixelCount = document.createElement("span");
  pixelCount.style.cssText = "margin-left:10px;color:#8b8b96;";

  wrapper.appendChild(size);
  wrapper.appendChild(pixelCount);
  return { wrapper, size, pixelCount };
}

app.registerExtension({
  name: "EnndeeResolutionSelector.Preview",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "Enndee_ResolutionSelector") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);

      const aspectWidget = this.widgets?.find((w) => w.name === "aspect_ratio");
      const megaWidget = this.widgets?.find((w) => w.name === "megapixels");
      const multipleWidget = this.widgets?.find((w) => w.name === "multiple");
      const keepWidget = this.widgets?.find((w) => w.name === "keep_source_aspect_ratio");
      const resizeTypeWidget = this.widgets?.find((w) => w.name === "resize_type");
      const multiplierWidget = this.widgets?.find((w) => w.name === "multiplier");
      const longerWidget = this.widgets?.find((w) => w.name === "longer_size");
      const shorterWidget = this.widgets?.find((w) => w.name === "shorter_size");
      if (!aspectWidget || !megaWidget || !multipleWidget) return;

      const { wrapper, size, pixelCount } = buildPreviewElement();
      this.addDOMWidget("resolution_preview", "preview", wrapper, {
        serialize: false,
        hideOnZoom: false,
        getValue: () => null,
        setValue: () => {},
      });
      wrapper.style.pointerEvents = "none";

      const refresh = () => {
        const result = computePreview(
          aspectWidget.value,
          Number(megaWidget.value) || 0,
          Number(multipleWidget.value) || 8,
          !!keepWidget?.value,
          !!this.inputs?.find((input) => input.name === "image")?.link,
          resizeTypeWidget?.value,
          {
            multiplier: Number(multiplierWidget?.value) || 0,
            longerSize: Number(longerWidget?.value) || 0,
            shorterSize: Number(shorterWidget?.value) || 0,
            megapixels: Number(megaWidget.value) || 0,
            multiple: Number(multipleWidget.value) || 0,
          },
        );
        size.textContent = result.text;
        size.style.fontStyle = result.dim ? "italic" : "normal";
        pixelCount.textContent = result.detail;
      };

      this.addEventListener?.("removed", () => {
        wrapper.remove();
      });

      [aspectWidget, megaWidget, multipleWidget, keepWidget, resizeTypeWidget,
       multiplierWidget, longerWidget, shorterWidget].forEach((widget) => {
        if (!widget) return;
        const original = widget.callback;
        widget.callback = function (value) {
          const res = original?.apply(this, arguments);
          refresh();
          return res;
        };
      });

      // Refresh the displayed preview when the source image is connected or removed.
      const onConnectionsChange = this.onConnectionsChange;
      this.onConnectionsChange = function () {
        const result = onConnectionsChange?.apply(this, arguments);
        setTimeout(refresh, 0);
        return result;
      };

      // also refresh when the node was loaded from a workflow
      const onConfigure = this.onConfigure;
      this.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        setTimeout(refresh, 0);
      };

      setTimeout(refresh, 0);
    };
  },
});