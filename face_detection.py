import numpy as np
import mss
import pyautogui
import pyray as pr
import face_alignment
import torch
from PIL import Image
import cv2
from scipy.spatial import ConvexHull

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device="mps",
    dtype=torch.bfloat16,
    face_detector="blazeface",
    face_detector_kwargs={"back_model": True},
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
        screen_img_np = cv2.resize(
            src=screen_img_np, dsize=(screen_width, screen_height)
        )
        screen_copy = screen_img_np.copy()
        screen_copy = cv2.blur(screen_copy, (27, 27))
        preds = fa.get_landmarks_from_image(screen_img_np)
        height, width, _ = screen_img_np.shape
        pr.begin_drawing()
        pr.clear_background(pr.BLANK)

        preds = preds if preds is not None else []
        for pred in preds:
            pred = pred if pred is not None else []
            contours = [(int(landmark[0]), int(landmark[1])) for landmark in pred]
            hull = ConvexHull(contours)
            vertex_pts = [contours[vertex] for vertex in hull.vertices]
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.array(vertex_pts), (255, 0, 0))
            face_blurred = cv2.bitwise_and(screen_copy, screen_copy, mask=mask)
            for pt in zip(mask.nonzero()[0], mask.nonzero()[1]):
                [r, g, b] = face_blurred[pt[1], pt[0]]
                pr.draw_circle(int(pt[1]), int(pt[0]), 1, pr.Color(r, g, b, 255))
            # for landmark in pred:
            #     x, y = landmark[0], landmark[1]
            #     pr.draw_circle(int(x), int(y), 2, pr.WHITE)
        pr.end_drawing()
    pr.close_window()


if __name__ == "__main__":
    main()
