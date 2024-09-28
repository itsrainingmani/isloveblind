import numpy as np
import mss
import pyautogui
import pyray as pr
from operator import itemgetter
import face_alignment
import torch
from PIL import Image

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
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        return img, screenshot.width, screenshot.height


def main():
    screen_width, screen_height = pyautogui.size()
    pr.set_config_flags(
        pr.ConfigFlags.FLAG_WINDOW_TRANSPARENT
        | pr.ConfigFlags.FLAG_WINDOW_TOPMOST
        | pr.ConfigFlags.FLAG_WINDOW_MOUSE_PASSTHROUGH
        | pr.ConfigFlags.FLAG_WINDOW_RESIZABLE
        | pr.ConfigFlags.FLAG_BORDERLESS_WINDOWED_MODE
        | pr.ConfigFlags.FLAG_WINDOW_HIGHDPI
    )
    pr.set_window_position(
        pr.get_monitor_width(0) // 2 - screen_width // 2,
        pr.get_monitor_height(0) // 2 - screen_height // 2,
    )
    pr.init_window(screen_width, screen_height, "Face Detection")
    pr.set_target_fps(60)
    print(screen_width, screen_height)

    while not pr.window_should_close():
        screen, capture_width, capture_height = capture_screen()
        preds = fa.get_landmarks_from_image(np.array(screen))

        scale_x = screen_width / capture_width
        scale_y = screen_height / capture_height

        pr.begin_drawing()
        pr.clear_background(pr.BLANK)

        if preds is not None:
            for pred in preds:
                if pred is not None:
                    for landmark in pred:
                        x, y = landmark[0], landmark[1]
                        x, y = x * scale_x, y * scale_y
                        pr.draw_circle(int(x), int(y), 2, pr.WHITE)

        pr.end_drawing()
    pr.close_window()


if __name__ == "__main__":
    main()
