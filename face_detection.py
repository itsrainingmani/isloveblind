import numpy as np
import mss
import pyautogui
from deepface import DeepFace
import pyray as pr
from operator import itemgetter


def detect_faces(image):
    faces = DeepFace.extract_faces(
        image,
        detector_backend="opencv",
        enforce_detection=False,
    )
    return faces


def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        screenshot = np.array(sct.grab(monitor))
        return screenshot, screenshot.shape[1], screenshot.shape[0]


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
        faces = detect_faces(screen)
        print(f"Detected {len(faces)} faces")

        scale_x = screen_width / capture_width
        scale_y = screen_height / capture_height

        pr.begin_drawing()
        pr.clear_background(pr.BLANK)

        for face in faces:
            if face["confidence"] < 0.9:
                continue
            params = face["facial_area"]
            x, y, w, h = itemgetter("x", "y", "w", "h")(params)

            # Scale coordinates
            scaled_x = int(x * scale_x)
            scaled_y = int(y * scale_y)
            scaled_w = int(w * scale_x)
            scaled_h = int(h * scale_y)

            print(
                f"Face detected at ({scaled_x}, {scaled_y}) with width {scaled_w} and height {scaled_h}"
            )
            pr.draw_rectangle_lines(scaled_x, scaled_y, scaled_w, scaled_h, pr.RED)

        pr.end_drawing()
    pr.close_window()


if __name__ == "__main__":
    main()
