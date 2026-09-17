"""
minimax_h3_promptor - MiniMax H3 Direct Promptor (Enndee), vendored.

This sub package contains the *MiniMax H3 Direct Promptor (Enndee)* node
(unified vision-LLM -> official-format H3 prompt generator), vendored from the
standalone pack ``ComfyUI-MiniMax-H3-Promptor-Enndee`` so that ComfyUI users
only need to install Enndees-Nodepack.

License: this component is **GPL-3.0** (fork of
https://github.com/1038lab/ComfyUI-Minimax-H3-Promptor) - see ``LICENSE`` in
this directory. The MIT license in the pack root applies only to Enndee's own
code, not to the code in this sub package.

On the first run a ``config.json`` is
auto-created from ``config.example.json`` (``config.json`` is git-ignored - it
holds your user API keys).
"""

from .py.h3_multimodal_promptor import H3_Multimodal_Promptor_Enndee

__all__ = ["H3_Multimodal_Promptor_Enndee"]