#  type: ignore

import math
from typing import Tuple, Union
import Quartz.CoreGraphics
import Quartz.QuartzCore
from PIL import Image
import Quartz
import mss
import numpy as np
import pyray as pr


def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def get_window_id(window_name):
    window_list = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
    )
    for window in window_list:
        if window_name in window.get("kCGWindowOwnerName", ""):
            return window["kCGWindowNumber"]
    return None


def capture_window(window_id):
    cgimg = Quartz.CGWindowListCreateImage(
        Quartz.CGRectNull,
        Quartz.kCGWindowListOptionIncludingWindow,
        window_id,
        Quartz.kCGWindowImageBoundsIgnoreFraming
        | Quartz.kCGWindowImageNominalResolution,
    )
    width = Quartz.CGImageGetWidth(cgimg)
    height = Quartz.CGImageGetHeight(cgimg)
    pixeldata = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(cgimg))
    bpr = Quartz.CGImageGetBytesPerRow(cgimg)
    # Convert to PIL Image.  Note: CGImage's pixeldata is BGRA
    image = Image.frombuffer("RGB", (width, height), pixeldata, "raw", "BGRX", bpr, 1)
    return np.array(image), width, height


def capture_screen_mss():
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        return img, screenshot.width, screenshot.height


def create_overlay_window(screen_width, screen_height):
    pr.set_config_flags(
        pr.ConfigFlags.FLAG_WINDOW_TRANSPARENT
        | pr.ConfigFlags.FLAG_WINDOW_TOPMOST
        | pr.ConfigFlags.FLAG_WINDOW_MOUSE_PASSTHROUGH
        | pr.ConfigFlags.FLAG_WINDOW_RESIZABLE
        | pr.ConfigFlags.FLAG_BORDERLESS_WINDOWED_MODE
        | pr.ConfigFlags.FLAG_WINDOW_HIGHDPI
        | pr.ConfigFlags.FLAG_VSYNC_HINT
    )
    pr.init_window(screen_width, screen_height, "is Love Blind?")
    pr.set_window_position(0, 0)
