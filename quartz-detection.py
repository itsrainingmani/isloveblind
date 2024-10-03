import time
import face_alignment.detection.blazeface
import face_alignment
import pyautogui
import torch
import pyray as pr
from windows import *
from graphics import *

fd = face_alignment.detection.blazeface.FaceDetector(device="mps")

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device="mps",
    dtype=torch.bfloat16,
    face_detector="blazeface",
)


def main():
    time.sleep(3)
    window_id = get_window_id("Edge")
    if window_id is None:
        print("Edge not found.")
        return

    screen_width, screen_height = pyautogui.size()
    create_overlay_window(screen_width, screen_height)

    while not pr.window_should_close():
        screen_img_np, capture_width, capture_height = capture_window(window_id)

        # preds = fa.get_landmarks_from_image(screen_img_np)
        preds = fd.detect_from_image(screen_img_np)

        scale_x = screen_width / capture_width
        scale_y = screen_height / capture_height
        pr.begin_drawing()
        pr.clear_background(pr.BLANK)
        preds = preds if preds is not None else []

        # draw_circle_on_face(preds)
        draw_ellipse_on_face(preds)
        # draw_blur_on_face(preds, screen_img_np)
        # draw_landmarks(preds, scale_x, scale_y)
        # blank_within_contour(preds, screen_img_np, scale_x, scale_y)

        pr.end_drawing()
    pr.close_window()


if __name__ == "__main__":
    main()
