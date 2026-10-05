"""Tests for the Meridian Prompt Composer node (conditional picture blocks)."""

import sys
import unittest
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR / "nodes"))

from meridian_prompt_composer import (  # noqa: E402
    DEFAULT_DESCRIPTIONS,
    MeridianPromptComposer,
)

DUMMY = object()  # any non-None value marks a connected image input


class MeridianPromptComposerTests(unittest.TestCase):
    def compose(self, **overrides):
        node = MeridianPromptComposer()
        kwargs = dict(
            prompt="A hero shot. {extra}",
            picture_descriptions=DEFAULT_DESCRIPTIONS,
            ref_image_1=None,
            ref_image_2=None,
            ref_image_3=None,
        )
        kwargs.update(overrides)
        (text,) = node.compose(**kwargs)
        return text

    def test_no_connected_pictures_leaves_the_prompt_untouched(self):
        self.assertEqual(self.compose(prompt="Just the base prompt."),
                         "Just the base prompt.")

    def test_connected_picture_inserts_its_block_at_the_placeholder(self):
        descriptions = ("ref_image_1:\n<Picture 2>: a red car\n"
                        "ref_image_2:\nref_image_3:\n")
        text = self.compose(picture_descriptions=descriptions, ref_image_1=DUMMY)
        self.assertEqual(text, "A hero shot. <Picture 2>: a red car")

    def test_block_is_appended_without_the_placeholder(self):
        descriptions = ("ref_image_1:\n<Picture 2>: a red car\n"
                        "ref_image_2:\nref_image_3:\n")
        text = self.compose(prompt="Base only.", picture_descriptions=descriptions,
                            ref_image_1=DUMMY)
        self.assertEqual(text, "Base only.\n\n<Picture 2>: a red car")

    def test_empty_block_is_skipped(self):
        text = self.compose(ref_image_1=DUMMY)  # DEFAULT_DESCRIPTIONS: empty blocks
        self.assertEqual(text, "A hero shot.")


if __name__ == "__main__":
    unittest.main()
