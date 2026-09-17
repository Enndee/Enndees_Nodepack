"""
enndee_colmap - vendored COLMAP / GLOMAP helper library for Enndees Nodepack.

This package contains the COLMAP binary parser and the COLMAP/GLOMAP CLI
wrappers.  They were originally taken from the MIT licensed project
``comfyui_colmap`` by nelsig (https://gitlab.com/nelsig/comfyui_colmap) and are
vendored here so that Enndees Nodepack is a self contained, all-in-one package
that does not depend on another custom node pack being installed.

Vendored / modified by Enndee:
  * removed hard coded developer paths (``D:\\OneDrive\\...``)
  * package relative imports (``enndee_colmap`` instead of the generic ``lib``)
  * ``max_image_size`` is now really forwarded to COLMAP's feature extractor
  * ``SiftMatching.use_gpu`` is now also passed for the sequential matcher
  * child processes no longer flash a console window on Windows
  * subprocess output is decoded with ``errors="replace"`` (no UnicodeDecodeError)
  * optional COLMAP 3.12+ global mapper backend (``colmap global_mapper``)

See NOTICE.md for the full attribution.
"""

from .colmap_parser import COLMAPParser
from .colmap_wrapper import COLMAPWrapper
from .glomap_wrapper import GLOMAPWrapper, fix_image_names_in_sparse

__all__ = [
    "COLMAPParser",
    "COLMAPWrapper",
    "GLOMAPWrapper",
    "fix_image_names_in_sparse",
]