# Enndees Nodepack

Custom nodes for **ComfyUI**: one-shot **Lichtfeld / 3DGS dataset creation** with
global Structure-from-Motion, plus a video frame extractor that also delivers the
matching audio.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)
![Platform](https://img.shields.io/badge/platform-Windows%20x64-informational)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-informational)

---

## Why this pack

Most camera-tracking nodes expect you to install COLMAP (and GLOMAP) by hand and
to type in absolute paths. This pack is **all-in-one**: the tested COLMAP and
GLOMAP builds are downloaded automatically into the pack's own `bin/` folder -
either by ComfyUI-Manager running `install.py` or lazily on the very first run.
Nothing is added to the system `PATH`, nothing has to be configured manually.

Combined with the integrated RMBG background removal, the `GLOMAP Lichtfeld
Tracker (Enndee)` node turns a frame sequence into a complete, ready-to-train
Lichtfeld Studio dataset in a single step.

---

## Nodes

| Node | ID | Purpose |
|------|----|---------|
| **GLOMAP Lichtfeld Tracker (Enndee)** | `Enndee_GLOMAPLichtfeldTracker` | Global SfM camera tracking + Lichtfeld dataset export |
| **Video Frame Extractor + Audio (Enndee)** | `Enndee_VideoFrameExtractorWithAudio` | Frame/audio extraction with an in-node timeline widget |
| **MiniMax H3 Direct Promptor (Enndee)** | `H3_Multimodal_Promptor_Enndee` | Official-format MiniMax H3 prompts from reference images (vision LLM) |
| **Resolution Selector (Enndee)** | `Enndee_ResolutionSelector` | Model-aware resolution presets + "keep source aspect ratio" sizing |

---

## Installation

### ComfyUI-Manager (recommended)

Install the pack as usual - Manager installs `requirements.txt` and then executes
`install.py`, which downloads COLMAP + GLOMAP into `<pack>/bin/`.
Watch the console for the `[Enndees-Nodepack]` messages.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone <your-repo-url> Enndees-Nodepack
cd Enndees-Nodepack
# uses the same python that runs ComfyUI, e.g.
python install.py
```

`install.py` is idempotent - running it again only downloads what is missing.

```text
python install.py                  # download what is missing
python install.py --check          # only report the current status
python install.py --force          # re-download everything
python install.py --nocuda         # CPU builds (no NVIDIA GPU)
python install.py --only glomap    # single component
python install.py --pin colmap="C:\Tools\colmap-x64-windows-cuda\COLMAP.bat"
python install.py --with-requirements
```

### Optional: RMBG background removal

The `use_rmbg` option of the GLOMAP node uses `transparent_background` (RMBG via
onnxruntime). When it is missing the node logs a warning and simply skips
background removal - everything else keeps working:

```bash
pip install transparent_background             # CPU onnxruntime is included
# optional: pip install onnxruntime-gpu        # CUDA acceleration (auto-detected)
```

### What gets downloaded

| Component | Version | Flavor | Size (approx.) | Why |
|-----------|---------|--------|----------------|-----|
| COLMAP | 3.11.1 | `cuda` / `nocuda` | 154 MB / 64 MB | SIFT features + matching |
| GLOMAP | 1.2.0 | `cuda` / `nocuda` | 74 MB / 18 MB | global mapper |

Both are official GitHub release archives of
[colmap/colmap](https://github.com/colmap/colmap/releases) and
[colmap/glomap](https://github.com/colmap/glomap/releases); checksums are
verified whenever upstream publishes them.

`auto` flavor = CUDA build when a CUDA GPU is detected, CPU build otherwise
(override with `--cuda` / `--nocuda` or `ENNDEE_BIN_FLAVOR`).

Everything lands in:

```text
Enndees-Nodepack/
|-- bin/                            # git-ignored
|   |-- colmap-3.11.1-cuda/
|   |-- glomap-1.2.0-cuda/
|   `-- enndee_binaries.json        # optional user config (pinned paths)
```

### Already have COLMAP/GLOMAP?

Nothing is downloaded when a usable binary is found. Resolution order:

1. the path typed into the node's `colmap_path` / `glomap_path` widget
2. `ENNDEE_COLMAP_PATH` / `ENNDEE_GLOMAP_PATH` environment variables
3. `bin/enndee_binaries.json` (created by `install.py --pin KIND=PATH`)
4. the pack's own `bin/` folder
5. auto detection (`PATH`, `C:\Tools\colmap*`, `C:\Program Files\COLMAP`, ...)

---

## Node: GLOMAP Lichtfeld Tracker (Enndee)

Runs the complete pipeline and writes a Lichtfeld Studio dataset:

```text
<lichtfeld_export_path>/
|-- images/           0001.png, 0002.png, ...   (full resolution)
|-- masks/            Lichtfeld / splat masks   (white = keep)
|-- masks_GLOMAP/     feature extraction masks  (white = excluded)
`-- sparse/0/         cameras.txt, images.txt, points3D.txt
```

### Inputs (required)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `colmap_path` | STRING | auto | COLMAP.bat / colmap.exe. Empty = auto detect / auto install |
| `glomap_path` | STRING | auto | glomap.exe (unused with `mapper_backend=colmap_global`) |
| `camera_model` | COMBO | `SIMPLE_PINHOLE` | intrinsics model (`PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`) |
| `matcher` | COMBO | `sequential` | `sequential` (video) or `exhaustive` (unordered images) |
| `max_features` | INT | 24000 | SIFT features per image |
| `images_path` | STRING | - | image folder, used when no `images` input is connected |
| `masks_path` | STRING | - | GLOMAP mask folder, used when no `masks_glomap` input |
| `lichtfeld_export_path` | STRING | - | export target; relative paths go below ComfyUI's `output/` |

### Inputs (optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `images` | IMAGE | - | frame batch (preferred source) |
| `masks_glomap` | MASK | - | masks for feature extraction (**white = excluded**) |
| `masks_lichtfeld` | MASK | - | splat masks for Lichtfeld (**white = keep**) |
| `use_rmbg` | BOOLEAN | true | built-in RMBG background removal |
| `rmbg_mode` | COMBO | `base` | `base`, `fast`, `base-nightly` |
| `rmbg_threshold` | FLOAT | 0.5 | segmentation sensitivity |
| `rmbg_resize` | COMBO | `static` | `static` or `dynamic` (up to 1280 px) |
| `use_gpu` | BOOLEAN | true | GPU for SIFT/RMBG |
| `keep_workspace` | BOOLEAN | false | keep the temporary COLMAP workspace |
| `auto_align` | BOOLEAN | true | align to the ground plane (Y up, floor at Y=0) |
| `sequential_overlap` | INT | 15 | neighbouring frames compared by the matcher |
| `max_image_size` | INT | 5120 | longest edge for feature extraction (export stays full res) |
| `frame_step` | INT | 2 | use every n-th frame |
| `downscale_factor` | FLOAT | 1.0 | SfM input scale only |
| `offset_glomap` | INT | 4 | erode (+) / dilate (-) the GLOMAP masks in px |
| `offset_splat` | INT | 12 | erode (+) / dilate (-) the Lichtfeld masks in px |
| `mapper_backend` | COMBO | `glomap` | `glomap` (GLOMAP 1.2.0) or `colmap_global` (COLMAP >= 3.12) |
| `auto_install_binaries` | BOOLEAN | true | download missing binaries on demand |
| `binary_flavor` | COMBO | `auto` | `auto`, `cuda`, `nocuda` |
| `embed_alpha_in_images` | BOOLEAN | false | also store the RMBG alpha in `images/` (RGBA PNGs) |

### Outputs

| Name | Type |
|------|------|
| `trajectory` | `CAMERA_TRAJECTORY` |
| `point_cloud` | `POINTCLOUD` |
| `confidence` | FLOAT |

### Mask conventions (important)

* `masks_glomap` / `masks_path`: **white (1) = region that is excluded from
  feature extraction** (typically the moving subject), black = features allowed.
  The masks are inverted internally because COLMAP expects white = valid.
* `masks_lichtfeld`: **white (1) = region to keep** in the splat. It is stored
  as-is in `masks/` (after `offset_splat`).
* RGBA images: the alpha channel is used automatically for `masks/` when no
  explicit Lichtfeld mask is connected.

### Recommended settings

* 24 fps 360 degree orbit of a subject: `matcher=sequential`,
  `sequential_overlap=15`, `frame_step=2`, `max_features=24000`,
  `max_image_size=5120`, `offset_glomap=+4`, `offset_splat=+12`.
* Unordered photo set: `matcher=exhaustive`, `frame_step=1`,
  `sequential_overlap` is ignored.
* Fast camera motion / shaky footage: increase `sequential_overlap` (20-30) and
  `max_features`, lower `frame_step` to 1.
* Very large images (> 4K): `downscale_factor=0.5` - the export stays full res.

---

## Node: Resolution Selector (Enndee)

Copied from the MIT licensed
[ComfyUI_ResolutionSelector](https://github.com/BRADSEC/ComfyUI_ResolutionSelector)
by BRADSEC (MIT, Copyright (c) 2023 BRADSEC) and extended with a source-aware
sizing mode.

Pick a model + preset resolution (the JS widget filters the list per model) and
get width, height and a matching empty latent. New in this pack:

* **`image` input** - connect a source image to derive the output size from it.
* **`keep_source_aspect_ratio`** - when enabled, the output size comes from the
  **aspect ratio of the connected image** and is scaled up *or* down to the
  requested megapixels instead of using the preset dropdown.
* **`target_megapixels`** - the pixel budget for that mode (`0` = use the
  megapixels of the selected preset resolution).

The result is always aligned to the model's `divisible_by` (e.g. 8 px, required
for a valid empty latent) and kept inside the model's min/max range; both axes
are scaled by the same factor, so the aspect ratio is preserved. The console
prints e.g. `source 1920x1080 -> 1328x744 (0.99 MP, target_megapixels=1.00 MP, aligned to 8px)`.
While `keep_source_aspect_ratio` is active the preset dropdown is dimmed in the
UI (it is not used in that mode).

### Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | COMBO | `SDXL` | Preset family: Flux, Qwen Image, Z-Image, SD 1.5, SDXL, All |
| `resolution` | COMBO | `1024x1024` | preset resolution (list filtered by `model`) |
| `resolution_multiplier` | COMBO | `1x` | `1x`-`4x`; scales the output (in aspect mode it scales the pixel budget) |
| `batch_size` | INT | 1 | latent batch size |
| `custom_width` / `custom_height` | INT | 0 | optional custom size (second output pair) |
| `custom_multiplier` / `custom_batch` | COMBO / INT | `1x` / 1 | multiplier and batch for the custom size |
| `image` | IMAGE | - | source image for the aspect-ratio mode |
| `keep_source_aspect_ratio` | BOOLEAN | false | size from the source aspect ratio |
| `target_megapixels` | FLOAT | 0.0 | pixel budget (0 = preset megapixels) |

### Outputs

| Name | Type |
|------|------|
| `width` / `height` | INT |
| `latent` | LATENT |
| `custom_width` / `custom_height` | INT |
| `custom_latent` | LATENT |

### Examples

| Source | Mode | Result |
|--------|------|--------|
| 1920x1080 | `target_megapixels=1.0` | 1328x744 (0.99 MP) |
| 640x480 | `target_megapixels=2.0` | 1632x1224 (2.00 MP) - upscaled |
| 1080x1920 | `target_megapixels=1.0` | 744x1328 (0.99 MP) - portrait |
| 3840x2160 | `target_megapixels=4.0`, model `All` | 2664x1496 (3.99 MP) |
| 1920x1080 | `target_megapixels=16.0`, model `SDXL` | 2048x1152 - clamped to the model maximum |

---

## Node: MiniMax H3 Direct Promptor (Enndee)

Generates **official-format MiniMax H3 prompts** directly from up to 8 reference
images (+ optional video keyframes) and your description - in a single vision-LLM
call (OpenAI, Ollama, Gemini or Claude). Vendored here from the standalone
`ComfyUI-MiniMax-H3-Promptor-Enndee` (GPL-3.0, fork of
[1038lab/ComfyUI-Minimax-H3-Promptor](https://github.com/1038lab/ComfyUI-Minimax-H3-Promptor)).

### Inputs (required)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | COMBO | T2V | Forces the H3 format: T2V, I2V, I2VA, V2V, V2VA, A2V, FL2VA, Ref2VA |
| `description` | STRING | - | your creative scene description |
| `duration` | FLOAT | 5.0 | video length in seconds (4-15) |

### Inputs (optional)

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_ref_1..8` | IMAGE | up to 8 reference images (mapped to `<Picture N>` labels) |
| `video_ref` | IMAGE | video batch - keyframes are extracted and mapped to `<Video 1>` |
| `output_language` | COMBO | `English` / `Chinese` |
| `provider` | COMBO | `openai`, `ollama`, `gemini`, `claude` |
| `api_key` | STRING | per-run override (config.json stays untouched) |
| `model_name` | COMBO | selected vision model (Ollama: lists your installed models) |
| `temperature` / `top_k` / `top_p` / `min_p` / `repeat_penalty` | - | sampling options (Ollama sent on every call) |
| `max_tokens` | INT | 4096 | LLM output budget (256-16384) |

### Output

| Name | Type |
|------|------|
| `prompt` | STRING |

### Configuration & models

* The selected model must be **vision-capable** (`gpt-4o`, Claude, Qwen-VL, Gemma
  vision, ...).
* On first use the node auto-creates `minimax_h3_promptor/config.json` from
  `config.example.json` - open it and add your API keys. The file is
  **git-ignored** and never shipped; you can also pass `api_key` per node run.
* Output is capped at MiniMax's official **7,000 character** limit
  (`max_tokens` is the generating LLM's budget, not H3's limit).

---

## License

* This pack's own code: **MIT** (see `LICENSE`).
* The copied `Enndee_ResolutionSelector` node is based on BRADSEC's MIT licensed
  [ComfyUI_ResolutionSelector](https://github.com/BRADSEC/ComfyUI_ResolutionSelector)
  (MIT, Copyright (c) 2023 BRADSEC) - MIT is compatible with this pack's license.
* The vendored `minimax_h3_promptor/` component: **GPL-3.0** (see
  `minimax_h3_promptor/LICENSE`) - a fork of the
  [1038lab/ComfyUI-Minimax-H3-Promptor](https://github.com/1038lab/ComfyUI-Minimax-H3-Promptor)
  project.