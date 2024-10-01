import time
import Quartz.CoreGraphics
import Quartz.QuartzCore
import numpy as np
import face_alignment
import pyautogui
import torch
from PIL import Image
import cv2
from scipy.spatial import ConvexHull
import Quartz
import pyray as pr

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device="mps",
    dtype=torch.bfloat16,
    face_detector="blazeface",
)


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
    return image, width, height


# image_np = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
# return Image.fromarray(image_np[:, :, :3]), width, height


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


def main():
    time.sleep(3)
    window_id = get_window_id("Edge")
    if window_id is None:
        print(f"Edge not found.")
        return

    screen_width, screen_height = pyautogui.size()
    create_overlay_window(screen_width, screen_height)

    while not pr.window_should_close():
        screen_img, capture_width, capture_height = capture_window(window_id)
        screen_img_np = np.array(screen_img)
        mask = np.zeros(screen_img_np.shape[:2], dtype=np.uint8)
        preds = fa.get_landmarks_from_image(screen_img_np)

        scale_x = screen_width / capture_width
        scale_y = screen_height / capture_height
        pr.begin_drawing()
        pr.clear_background(pr.BLANK)

        preds = preds if preds is not None else []
        for pred in preds:
            pred = pred if pred is not None else []
            contours = [
                (int(landmark[0] * scale_x), int(landmark[1] * scale_y))
                for landmark in pred
            ]
            # for c in contours:
            #     pr.draw_circle(int(c[0] * scale_x), int(c[1] * scale_y), 2, pr.WHITE)
            hull = ConvexHull(contours)
            vertex_pts = [contours[vertex] for vertex in hull.vertices]
            cv2.fillConvexPoly(mask, np.array(vertex_pts), color=(255, 0, 0))
            zipapped = zip(*np.nonzero(mask))
            for pt in zipapped:
                pr.draw_circle(int(pt[1] * scale_x), int(pt[0] * scale_y), 1, pr.WHITE)
            # blurred = cv2.GaussianBlur(mask, (25, 25), 0)
            # non_zero_indices = np.nonzero(blurred)
            # for y, x in zip(*non_zero_indices):
            #     color = blurred[y, x]
            #     pr.draw_pixel(
            #         int(x * scale_x),
            #         int(y * scale_y),
            #         pr.Color(color, color, color, 255),
            #     )

        pr.draw_fps(0, 0)
        pr.end_drawing()
    pr.close_window()


if __name__ == "__main__":
    main()
