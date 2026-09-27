"""
Enndees Nodepack - Custom Nodes for ComfyUI
GLOMAP Camera Tracker with advanced mask handling for Lichtfeld Studio.
"""

import sys
import os

print("\033[96m[Enndee] Enndees Nodepack loaded\033[0m")

# Add this directory and nodes/ to path for imports.
# The COLMAP/GLOMAP helpers are vendored (enndee_colmap/) so the pack is
# fully self contained - no external comfyui_colmap is needed.
_pack_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _pack_dir)
sys.path.insert(0, os.path.join(_pack_dir, "nodes"))

try:
    from glomap_lichtfeld_node import GLOMAPLichtfeldTracker
except Exception as _e:
    print(f"\033[31m[Enndee] GLOMAPLichtfeldTracker unavailable: {_e}\033[0m")
    GLOMAPLichtfeldTracker = None

try:
    from enndee_video_frame_extractor import Enndee_VideoFrameExtractorWithAudio
except Exception as _e:
    print(f"\033[31m[Enndee] VideoFrameExtractorWithAudio unavailable: {_e}\033[0m")
    Enndee_VideoFrameExtractorWithAudio = None

try:
    from minimax_h3_promptor import H3_Multimodal_Promptor_Enndee
except Exception as _e:
    print(f"\033[31m[Enndee] H3_Multimodal_Promptor_Enndee unavailable: {_e}\033[0m")
    H3_Multimodal_Promptor_Enndee = None

try:
    from enndee_resolution_selector import ResolutionSelectorEnndee
except Exception as _e:
    print(f"\033[31m[Enndee] ResolutionSelectorEnndee unavailable: {_e}\033[0m")
    ResolutionSelectorEnndee = None

try:
    from enndee_image_loader import ImageLoaderResizeEnndee
except Exception as _e:
    print(f"\033[31m[Enndee] ImageLoaderResizeEnndee unavailable: {_e}\033[0m")
    ImageLoaderResizeEnndee = None

try:
    from enndee_meridian_parameter_picker import MeridianParameterPickerEnndee
except Exception as _e:
    print(f"\033[31m[Enndee] MeridianParameterPickerEnndee unavailable: {_e}\033[0m")
    MeridianParameterPickerEnndee = None

try:
    from enndee_meridian_camera_path import MeridianCameraPathConfigurator
except Exception as _e:
    print(f"\033[31m[Enndee] MeridianCameraPathConfigurator unavailable: {_e}\033[0m")
    MeridianCameraPathConfigurator = None

try:
    from enndee_meridian_geometry import EnndeeMeridianGeometry
except Exception as _e:
    print(f"\033[31m[Enndee] EnndeeMeridianGeometry unavailable: {_e}\033[0m")
    EnndeeMeridianGeometry = None

try:
    from enndee_meridian_fast_depth import EnndeeMeridianFastDepth
except Exception as _e:
    print(f"\033[31m[Enndee] EnndeeMeridianFastDepth unavailable: {_e}\033[0m")
    EnndeeMeridianFastDepth = None

try:
    from lichtfeld_training_node import LichtfeldHeadlessTrainer
except Exception as _e:
    print(f"\033[31m[Enndee] LichtfeldHeadlessTrainer unavailable: {_e}\033[0m")
    LichtfeldHeadlessTrainer = None

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if GLOMAPLichtfeldTracker is not None:
    NODE_CLASS_MAPPINGS["Enndee_GLOMAPLichtfeldTracker"] = GLOMAPLichtfeldTracker
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_GLOMAPLichtfeldTracker"] = "GLOMAP Lichtfeld Tracker (Enndee)"

# WEB_DIRECTORY lets ComfyUI serve static files (web/js/...).
# The timeline widget for Enndee_VideoFrameExtractorWithAudio lives there.
if Enndee_VideoFrameExtractorWithAudio is not None:
    NODE_CLASS_MAPPINGS["Enndee_VideoFrameExtractorWithAudio"] = Enndee_VideoFrameExtractorWithAudio
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_VideoFrameExtractorWithAudio"] = "Video Frame Extractor + Audio (Enndee)"

if H3_Multimodal_Promptor_Enndee is not None:
    NODE_CLASS_MAPPINGS["H3_Multimodal_Promptor_Enndee"] = H3_Multimodal_Promptor_Enndee
    NODE_DISPLAY_NAME_MAPPINGS["H3_Multimodal_Promptor_Enndee"] = "MiniMax H3 Direct Promptor (Enndee)"

if ResolutionSelectorEnndee is not None:
    NODE_CLASS_MAPPINGS["Enndee_ResolutionSelector"] = ResolutionSelectorEnndee
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_ResolutionSelector"] = "Resolution Selector (Enndee)"

if ImageLoaderResizeEnndee is not None:
    NODE_CLASS_MAPPINGS["Enndee_ImageLoaderResize"] = ImageLoaderResizeEnndee
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_ImageLoaderResize"] = "Load & Resize Image (Enndee)"

if MeridianParameterPickerEnndee is not None:
    NODE_CLASS_MAPPINGS["Enndee_MeridianParameterPicker"] = MeridianParameterPickerEnndee
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_MeridianParameterPicker"] = "Meridian Parameter Picker (Enndee)"

if MeridianCameraPathConfigurator is not None:
    NODE_CLASS_MAPPINGS["Enndee_MeridianCameraPath"] = MeridianCameraPathConfigurator
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_MeridianCameraPath"] = "Meridian Camera Path Configurator (Enndee)"

if EnndeeMeridianGeometry is not None:
    NODE_CLASS_MAPPINGS["Enndee_MeridianGeometry"] = EnndeeMeridianGeometry
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_MeridianGeometry"] = "Meridian Geometry (Enndee)"

if EnndeeMeridianFastDepth is not None:
    NODE_CLASS_MAPPINGS["Enndee_MeridianFastDepthEstimator"] = EnndeeMeridianFastDepth
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_MeridianFastDepthEstimator"] = "Meridian Fast Depth Estimator (Enndee)"

if LichtfeldHeadlessTrainer is not None:
    NODE_CLASS_MAPPINGS["Enndee_LichtfeldHeadlessTrainer"] = LichtfeldHeadlessTrainer
    NODE_DISPLAY_NAME_MAPPINGS["Enndee_LichtfeldHeadlessTrainer"] = "Lichtfeld Headless Trainer (Enndee)"

WEB_DIRECTORY = os.path.join(_pack_dir, "web")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]