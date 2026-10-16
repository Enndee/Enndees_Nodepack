# SPDX-License-Identifier: MIT
"""Compose the Meridian i2v prompt conditionally from the connected reference pictures.

The H3 tokenizer only numbers the pictures that are actually connected (`<Picture i>`
counts emitted references in order), but the prompt text is a static widget - and the
model cannot be told "if a picture is connected...", because that is just text to it.
This node makes the text follow the graph instead. Two fields:

- `prompt` - the basic prompt; `<Picture 1>` is described here as usual. `{extra}` marks
  where the connected ref_image blocks are inserted (appended at the end when the
  placeholder is missing);
- `picture_descriptions` - one block per extra picture, split by `ref_image_1:` /
  `ref_image_2:` / `ref_image_3:` header lines, describing `<Picture 2>` / `<Picture 3>` /
  `<Picture 4>` respectively (picture labels count connection order).

A block is inserted only while its input is connected; empty blocks are skipped, so wiring
a photo without text changes nothing. Wire each extra photo to BOTH the reference node's
`ref_images.ref_image_k` AND this node's `ref_image_k` (same photo, same order, no gaps -
the node warns about gaps and about blocks whose `<Picture n>` tag does not match).

Why one text block instead of one widget per picture: ComfyUI stores a node's widget
values BY POSITION, so every time this node's widget list changed, previously saved
workflows shifted their texts into the wrong boxes (and the frontend sometimes promoted
the stale fields to input slots). A single block with explicit headers cannot shift.

Graph setup: right-click the reference node's prompt widget -> "Convert widget to
input", then wire this node's `prompt` output into it (v7 already does). Bypass this node
to fall back to the plain prompt widget.
"""

import re

DEFAULT_DESCRIPTIONS = "ref_image_1:\n\nref_image_2:\n\nref_image_3:\n"



class MeridianPromptComposer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "subject_definitions:\n<Picture 1>: Source photo establishing the visual identity, faces, hair, clothing, props, background environment, materials, and lighting of the scene.\n<Video 1>: Layout and camera trajectory reference establishing the exact framing, continuous camera path, and spatial composition. Flat mid-grey areas represent unobserved spatial holes to be filled.\n{extra}\n\nsummary:\n[video editing + viewpoint change] The target video presents the exact scene from <Picture 1> in a frozen-in-time bullet-time state. Time is completely stopped at the precise moment captured in <Picture 1>; every person, pose, expression, piece of clothing, prop, background element, and light source remains 100% static and motionless. The target video completes <Video 1> by following its exact camera trajectory and framing while seamlessly filling all mid-grey occlusion holes with content matching <Picture 1>.\n\nretention_analysis:\n<Picture 1> (source photo): fully_preserved - the subjects, facial features, poses, hair, clothing, lighting, props, and environment are fully preserved in a completely frozen state. No subject motion, restyling, or addition occurs.\n<Video 1> (layout reference): fully_preserved - the camera path, framing, shot scale, and camera timing are strictly maintained frame-by-frame; mid-grey occlusion holes are filled in and point-cloud artifacts are cleaned up.\n\ndetailed_description:\n[Shot 1] One continuous shot framing the scene of <Picture 1>, following the exact frame-by-frame camera movement and composition of <Video 1>. Time is completely frozen: all characters, facial expressions, hair strands, fabric folds, props, and environmental background elements remain entirely static and locked in their instant freeze-frame poses throughout the entire shot. Zero subject movement, breathing, background drift, or dynamic scene animation occurs. The camera performs the smooth movement defined by <Video 1>, revealing the static 3D scene from the new viewpoint while filling all mid-grey occlusion holes with seamless content consistent with <Picture 1>. There are no cuts.\n\noverall_soundscape:\nNo ambient environmental sound, no dialogue, and no diegetic sound effects.\n\nnon_diegetic_music:\nNo music.",
                                      "tooltip": "Basic prompt (<Picture 1> is described here). "
                                                 "`{extra}` marks where the connected ref_image "
                                                 "blocks are inserted."}),
                "picture_descriptions": ("STRING", {"multiline": True,
                                                    "default": "ref_image_1:\n<Picture 2> is a further photo of the same scene from another angle: the same subjects, materials, props and lighting as <Picture 1>, only seen from elsewhere. It is not a new location and adds no new content; use it to disambiguate what is behind, beside or hidden in <Picture 1>. Where it disagrees with <Video 1> about camera path or framing, <Video 1> still decides.\n\nref_image_2:\n<Picture 3> is a further photo of the same scene from another angle: the same subjects, materials, props and lighting as <Picture 1>, only seen from elsewhere. It is not a new location and adds no new content; use it to disambiguate what is behind, beside or hidden in <Picture 1>. Where it disagrees with <Video 1> about camera path or framing, <Video 1> still decides.\n\nref_image_3:\n<Picture 4> is a further photo of the same scene from another angle: the same subjects, materials, props and lighting as <Picture 1>, only seen from elsewhere. It is not a new location and adds no new content; use it to disambiguate what is behind, beside or hidden in <Picture 1>. Where it disagrees with <Video 1> about camera path or framing, <Video 1> still decides.",
                                                    "tooltip": "One block per extra picture, split "
                                                               "by `ref_image_1:` / `ref_image_2:` "
                                                               "/ `ref_image_3:` headers. A block is "
                                                               "inserted only while its input is "
                                                               "connected; it should describe "
                                                               "<Picture 2> / <Picture 3> / "
                                                               "<Picture 4>."}),
            },
            "optional": {
                "ref_image_1": ("IMAGE", {"tooltip": "Connect the SAME photo as "
                                                     "ref_images.ref_image_1 on the reference "
                                                     "node - it is <Picture 2>."}),
                "ref_image_2": ("IMAGE", {"tooltip": "Same photo as ref_images.ref_image_2 - "
                                                     "it is <Picture 3>."}),
                "ref_image_3": ("IMAGE", {"tooltip": "Same photo as ref_images.ref_image_3 - "
                                                     "it is <Picture 4>."}),
            },
        }


    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "compose"
    CATEGORY = "conditioning"
    DESCRIPTION = (
        "Adds per-picture paragraphs to the prompt only while that picture input is "
        "connected. All paragraphs live in one `picture_descriptions` block split by "
        "`ref_image_1/2/3:` headers, so re-saved workflows can never shift them into the "
        "wrong box."
    )

    @staticmethod
    def _sections(text):
        """Split the `ref_image_k:` blocks: {'1': text, ...} ('' when a block is empty)."""
        sections, current = {1: [], 2: [], 3: []}, None
        for line in text.splitlines():
            header = re.match(r"\s*ref_image_([123])\s*:\s*$", line)
            if header:
                current = int(header.group(1))
                continue
            if current is not None:
                sections[current].append(line)
        return {key: "\n".join(lines).strip() for key, lines in sections.items()}

    def compose(self, prompt, picture_descriptions, ref_image_1=None, ref_image_2=None,
                ref_image_3=None):
        connected = {1: ref_image_1 is not None, 2: ref_image_2 is not None,
                     3: ref_image_3 is not None}
        sections = self._sections(picture_descriptions)
        if picture_descriptions.strip() and not any(sections.values()):
            print("MeridianPromptComposer: WARNING - `picture_descriptions` has no "
                  "`ref_image_1:`/`ref_image_2:`/`ref_image_3:` header lines, so nothing can be "
                  "inserted")
        if not connected[1] and (connected[2] or connected[3]):
            print("MeridianPromptComposer: WARNING - a gap in the connected pictures shifts the "
                  "<Picture i> labels; connect ref_image_1 (and ref_image_2) before the later ones")
        blocks, included, skipped = [], [], []
        for key in (1, 2, 3):
            text = sections[key]
            if not text:
                continue
            match = re.match(r"<Picture (\d+)>", text)
            if match and int(match.group(1)) != key + 1:
                print(f"MeridianPromptComposer: WARNING - the `ref_image_{key}:` block starts with "
                      f"<Picture {match.group(1)}> but should describe <Picture {key + 1}>")
            if connected[key]:
                blocks.append(text)
                included.append(f"ref_image_{key} (<Picture {key + 1}>)")
            else:
                skipped.append(f"ref_image_{key} (<Picture {key + 1}>, not connected)")
        extra = "\n".join(blocks)
        if "{extra}" in prompt:
            text = re.sub(r"\n{3,}", "\n\n", prompt.replace("{extra}", extra)).strip()
        elif extra:
            text = prompt.rstrip() + "\n\n" + extra
        else:
            text = prompt.strip()
        summary = "included " + (", ".join(included) if included else "no extra descriptions")
        if skipped:
            summary += " | skipped " + ", ".join(skipped)
        print(f"MeridianPromptComposer: {summary} -> {len(text)} chars")
        return (text,)


NODE_CLASS_MAPPINGS = {"MeridianPromptComposer": MeridianPromptComposer}
NODE_DISPLAY_NAME_MAPPINGS = {"MeridianPromptComposer": "Meridian Prompt Composer (conditional pictures)"}
