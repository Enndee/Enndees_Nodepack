# NOTICE / Attribution
#
# The files in this folder were taken from the MIT licensed project
# `comfyui_colmap` (a.k.a. "ComfyUI NELSIG Tools") by nelsig:
#
#     https://gitlab.com/nelsig/comfyui_colmap
#     Author: nelsig - https://www.nelsig.ch
#     License: MIT
#
# Vendored files and their changes:
#
#   colmap_parser.py   - byte for byte identical, only this header was added.
#   colmap_wrapper.py  - hard coded developer paths removed, console window
#                        suppressed on Windows, subprocess output decoded with
#                        errors="replace", SiftMatching.use_gpu forwarded for the
#                        sequential matcher, configurable timeouts, helper to
#                        detect COLMAP >= 3.12 sub commands.
#   glomap_wrapper.py  - hard coded developer paths removed, max_image_size is
#                        now really forwarded to the feature extractor, optional
#                        `colmap global_mapper` backend (COLMAP >= 3.12),
#                        console window suppressed, configurable timeout.
#
# Everything else (packaging, path resolution, automatic binary download,
# Lichtfeld dataset export, node logic) was written for Enndees Nodepack.
#
# MIT License of the original work:
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.