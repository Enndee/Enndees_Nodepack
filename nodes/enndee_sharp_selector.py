"""Sharpness Analyzer + Top-N Sharp Frame Selector (Enndee).

Self-contained, extended version of the *Sharpness Analyzer* / *Sharp Frame
Selector* duo from ComfyUI-Sharp-Selector (MIT). Frames are scored with the
Laplacian variance (higher = sharper) and the batch is reduced to its sharpest
frames.

The selector adds a ``selection_method`` option the original node lacks:

* ``batched_topn`` (default) - keep the **top ``num_frames`` frames of every
  ``batch_size`` chunk**, e.g. the 3 sharpest of every 4 frames.
* ``batched``  - original behaviour: the single sharpest frame of every chunk.
* ``best_n``   - the globally sharpest ``num_frames`` frames.

``batch_buffer`` skips frames between chunks (stride = batch_size + buffer); 0
gives contiguous coverage. ``min_sharpness`` drops frames below the threshold
(0.0 keeps everything). The ``SHARPNESS_SCORES`` type is shared with
ComfyUI-Sharp-Selector, so the Enndee analyzer/selector are interchangeable
with the original pair.
"""

import numpy as np
import torch

_TAG = "[Enndee/SharpSelector]"


def _log(message):
    print("{} {}".format(_TAG, message), flush=True)


class Enndee_SharpnessAnalyzer:
    """Laplacian-variance sharpness score for every frame of an IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE", {
            "tooltip": "Batch of frames to score (e.g. an extracted video "
                       "clip). One Laplacian-variance score per frame - "
                       "higher means sharper."})}}

    RETURN_TYPES = ("SHARPNESS_SCORES",)
    RETURN_NAMES = ("scores",)
    FUNCTION = "analyze_sharpness"
    CATEGORY = "Enndee/image"
    DESCRIPTION = ("Calculate the Laplacian variance (sharpness) of every frame "
                   "in an IMAGE batch and output the scores for a Sharp Frame "
                   "Selector node.")

    def analyze_sharpness(self, images):
        try:
            import cv2
        except Exception as exc:  # cv2 ships with ComfyUI
            raise RuntimeError("Sharpness Analyzer (Enndee) needs OpenCV "
                               "(cv2): {}".format(exc))

        _log("Scoring {} frames...".format(len(images)))
        scores = []
        for index in range(len(images)):
            frame = images[index].cpu().numpy()
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        return (scores,)


class Enndee_SharpFrameSelector:
    """Reduce an IMAGE batch to its sharpest frames (per chunk or globally)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {
                "tooltip": "The full frame batch to reduce (the same batch fed "
                           "to the Sharpness Analyzer)."}),
            "scores": ("SHARPNESS_SCORES", {
                "tooltip": "Scores from a Sharpness Analyzer node (Enndee or "
                           "ComfyUI-Sharp-Selector)."}),
            "selection_method": (["batched_topn", "batched", "best_n"], {
                "default": "batched_topn",
                "tooltip": "batched_topn: keep the top num_frames of every "
                           "batch_size chunk (e.g. 3 of every 4). batched: keep "
                           "only the single sharpest frame of every chunk. "
                           "best_n: keep the globally sharpest num_frames "
                           "frames."}),
            "batch_size": ("INT", {
                "default": 24, "min": 1, "max": 100000, "step": 1,
                "tooltip": "Chunk length in frames. Use 4 to group the clip "
                           "into batches of four frames."}),
            "batch_buffer": ("INT", {
                "default": 0, "min": 0, "max": 100000, "step": 1,
                "tooltip": "Frames skipped between chunks (stride = batch_size "
                           "+ batch_buffer). Keep 0 for contiguous, gap-free "
                           "coverage of the whole clip."}),
            "num_frames": ("INT", {
                "default": 10, "min": 1, "max": 100000, "step": 1,
                "tooltip": "How many frames to keep: per chunk in batched_topn "
                           "(e.g. 3) and globally in best_n. Ignored by "
                           "batched."}),
            "min_sharpness": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.1,
                "tooltip": "Drop frames scoring below this Laplacian-variance "
                           "value. 0.0 keeps everything."}),
        }}

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("selected_images", "count")
    FUNCTION = "select_frames"
    CATEGORY = "Enndee/image"
    DESCRIPTION = (
        "Reduce an IMAGE batch to its sharpest frames. batched_topn keeps the "
        "top num_frames of every batch_size chunk (e.g. the 3 sharpest of every "
        "4 frames); batched keeps one per chunk; best_n keeps the global top N.")

    def select_frames(self, images, scores, selection_method, batch_size,
                      batch_buffer, num_frames, min_sharpness):
        batch_size = max(1, int(batch_size))
        batch_buffer = max(0, int(batch_buffer))
        num_frames = max(1, int(num_frames))
        min_sharpness = float(min_sharpness)

        if len(scores) != len(images):
            limit = min(len(images), len(scores))
            _log("images ({}) and scores ({}) differ - using the first {} "
                 "frames.".format(len(images), len(scores), limit))
            images = images[:limit]
            scores = scores[:limit]

        if selection_method == "batched_topn":
            selected = self._select_batched_topn(
                scores, batch_size, batch_buffer, num_frames, min_sharpness)
        elif selection_method == "batched":
            selected = self._select_batched(
                scores, batch_size, batch_buffer, min_sharpness)
        else:  # "best_n"
            selected = self._select_best_n(scores, num_frames, min_sharpness)

        if not selected:
            _log("No frame passed min_sharpness={} - returning an empty "
                 "placeholder.".format(min_sharpness))
            return self._empty_result(images)

        _log("Selected {} of {} frames with '{}'.".format(
            len(selected), len(images), selection_method))
        return (images[selected], len(selected))

    @staticmethod
    def _select_batched_topn(scores, batch_size, batch_buffer, num_frames,
                             min_sharpness):
        """Keep the top ``num_frames`` of every ``batch_size`` chunk."""
        selected = []
        step = batch_size + batch_buffer
        for start in range(0, len(scores), step):
            chunk = np.asarray(scores[start:start + batch_size], dtype=np.float64)
            if chunk.size == 0:
                continue
            keep = min(num_frames, chunk.size)
            top_local = np.argsort(chunk)[-keep:]
            for local in sorted(int(i) for i in top_local):
                if float(chunk[local]) >= min_sharpness:
                    selected.append(start + local)
        return selected

    @staticmethod
    def _select_batched(scores, batch_size, batch_buffer, min_sharpness):
        """Original behaviour: the single sharpest frame of every chunk."""
        selected = []
        step = batch_size + batch_buffer
        for start in range(0, len(scores), step):
            chunk = np.asarray(scores[start:start + batch_size], dtype=np.float64)
            if chunk.size == 0:
                continue
            best = int(np.argmax(chunk))
            if float(chunk[best]) >= min_sharpness:
                selected.append(start + best)
        return selected

    @staticmethod
    def _select_best_n(scores, num_frames, min_sharpness):
        """Keep the globally sharpest ``num_frames`` frames."""
        valid = [i for i, score in enumerate(scores)
                 if float(score) >= min_sharpness]
        if not valid:
            return []
        valid_scores = np.asarray([float(scores[i]) for i in valid],
                                  dtype=np.float64)
        keep = min(num_frames, len(valid))
        top_local = np.argsort(valid_scores)[-keep:]
        return sorted(valid[int(i)] for i in top_local)

    @staticmethod
    def _empty_result(images):
        if len(images) == 0:
            return (images, 0)
        height, width = int(images.shape[1]), int(images.shape[2])
        channels = int(images.shape[-1])
        empty = torch.zeros((1, height, width, channels),
                            dtype=images.dtype, device=images.device)
        return (empty, 0)


NODE_CLASS_MAPPINGS = {
    "Enndee_SharpnessAnalyzer": Enndee_SharpnessAnalyzer,
    "Enndee_SharpFrameSelector": Enndee_SharpFrameSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Enndee_SharpnessAnalyzer": "Sharpness Analyzer (Enndee)",
    "Enndee_SharpFrameSelector": "Sharp Frame Selector Top-N (Enndee)",
}

__all__ = [
    "Enndee_SharpnessAnalyzer",
    "Enndee_SharpFrameSelector",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]


