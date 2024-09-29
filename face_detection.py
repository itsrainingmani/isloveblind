import numpy as np
import mss
import pyautogui
import pyray as pr
import face_alignment
import raylib
import torch
from PIL import Image
import cv2
from scipy.spatial import ConvexHull
import itertools

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device="mps",
    dtype=torch.bfloat16,
    face_detector="blazeface",
)


def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        screenshot = sct.grab(monitor)
        return screenshot, screenshot.width, screenshot.height


def main():
    screen_width, screen_height = pyautogui.size()
    pr.set_config_flags(
        pr.ConfigFlags.FLAG_WINDOW_TRANSPARENT
        | pr.ConfigFlags.FLAG_WINDOW_TOPMOST
        | pr.ConfigFlags.FLAG_WINDOW_MOUSE_PASSTHROUGH
        | pr.ConfigFlags.FLAG_WINDOW_RESIZABLE
        | pr.ConfigFlags.FLAG_BORDERLESS_WINDOWED_MODE
        | pr.ConfigFlags.FLAG_WINDOW_HIGHDPI
        | pr.ConfigFlags.FLAG_VSYNC_HINT
    )
    pr.set_window_position(
        pr.get_monitor_width(0) // 2 - screen_width // 2,
        pr.get_monitor_height(0) // 2 - screen_height // 2,
    )
    pr.init_window(screen_width, screen_height, "Face Detection")
    print(screen_width, screen_height)

    while not pr.window_should_close():
        screenshot, capture_width, capture_height = capture_screen()

        # convert RGBA image to a PIL RGB Image
        screen_img = Image.frombytes(
            "RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"
        )
        screen_img_np = np.array(screen_img)
        preds = fa.get_landmarks_from_image(screen_img_np)

        scale_x = screen_width / capture_width
        scale_y = screen_height / capture_height

        pr.begin_drawing()
        pr.clear_background(pr.BLANK)

        preds = preds if preds is not None else []
        for pred in preds:
            pred = pred if pred is not None else []
            # landmarks = list(
            #     itertools.chain(pred[1:17], pred[27:23:-1], pred[22:18:-1])
            # )
            contours = [
                (int(landmark[0] * scale_x), int(landmark[1] * scale_y))
                for landmark in pred
            ]
            mask = np.zeros(screen_img_np.shape[:2], dtype=np.uint8)
            hull = ConvexHull(contours)
            vertex_pts = [contours[vertex] for vertex in hull.vertices]
            cv2.fillConvexPoly(mask, np.array(vertex_pts), color=(255, 0, 0))
            # blurred = cv2.GaussianBlur(mask, (25, 25), 0)
            # screen_img_np = np.where(
            #     mask[:, :, None] == 255, blurred[:, :, None], screen_img_np
            # )
            zipapped = zip(mask.nonzero()[0], mask.nonzero()[1])
            for pt in zipapped:
                pr.draw_circle(int(pt[1]), int(pt[0]), 2, pr.WHITE)
            # for pt in mask.nonzero():
            #     pr.draw_circle(int(pt[0]), int(pt[1]), 2, pr.WHITE)
            # for landmark in pred:
            #     x, y = landmark[0], landmark[1]
            #     x, y = x * scale_x, y * scale_y
            #     pr.draw_circle(int(x), int(y), 2, pr.WHITE)

        # texture = pr.load_texture_from_image(
        #     pr.Image(
        #         screen_img_np,
        #         screen_width,
        #         screen_height,
        #         1,
        #         pr.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8,
        #     )
        # )
        pr.begin_drawing()
        pr.clear_background(pr.BLANK)
        # pr.draw_texture(texture, 0, 0, pr.WHITE)
        pr.end_drawing()
        # pr.unload_texture(texture)
    pr.close_window()


if __name__ == "__main__":
    main()
