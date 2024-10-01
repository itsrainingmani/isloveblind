import math
import time
from typing import Tuple, Union
import cv2
import numpy as np
import mss
import pyautogui
import pyray as pr
from PIL import Image
import mediapipe as mp

model_path = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=4,
    min_face_detection_confidence=0.4,
    min_tracking_confidence=0.4,
    output_face_blendshapes=False,
)


def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


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
        | pr.ConfigFlags.FLAG_VSYNC_HINT
    )
    pr.set_window_position(
        pr.get_monitor_width(0) // 2 - screen_width // 2,
        pr.get_monitor_height(0) // 2 - screen_height // 2,
    )
    pr.init_window(screen_width, screen_height, "Face Detection")
    print(screen_width, screen_height)

    with FaceLandmarker.create_from_options(options) as landmarker:
        while not pr.window_should_close():
            screen, capture_width, capture_height = capture_screen()
            screen_np = np.array(screen)
            frame_copy = screen_np.copy()
            frame_copy = cv2.blur(frame_copy, (27, 27))
            height, width, _ = screen_np.shape
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=screen_np)
            face_landmarks_result = landmarker.detect(mp_image)
            face_landmarks_list = face_landmarks_result.face_landmarks

            scale_x = screen_width / capture_width
            scale_y = screen_height / capture_height

            pr.begin_drawing()
            pr.clear_background(pr.BLANK)

            for face in face_landmarks_list:
                # facial_landmarks = []
                # for landmark in face:
                #     x = landmark.x
                #     y = landmark.y
                #     x, y = normalized_to_pixel_coordinates(
                #         x, y, screen_width, screen_height
                #     )
                #     facial_landmarks.append([x, y])
                # convexhull = cv2.convexHull(np.array(facial_landmarks))
                # mask = np.zeros((height, width), np.uint8)
                # cv2.fillConvexPoly(mask, convexhull, (255, 0, 0))
                # face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
                # for pt in zip(mask.nonzero()[0], mask.nonzero()[1]):
                #     [r, g, b] = face_extracted[pt[1], pt[0]]
                #     pr.draw_circle(int(pt[1]), int(pt[0]), 1, pr.Color(r, g, b, 255))
                for landmarks in face:
                    x, y = landmarks.x, landmarks.y
                    x, y = normalized_to_pixel_coordinates(
                        x, y, screen_width, screen_height
                    )
                    pr.draw_circle(int(x), int(y), 2, pr.WHITE)

            pr.end_drawing()
        pr.close_window()


if __name__ == "__main__":
    main()
