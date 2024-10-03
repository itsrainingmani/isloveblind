import time
import cv2
import pyautogui
import pyray as pr
import mediapipe as mp

from windows import (
    capture_window,
    create_overlay_window,
    get_window_id,
    normalized_to_pixel_coordinates,
)

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


def main():
    time.sleep(3)
    window_id = get_window_id("Edge")
    if window_id is None:
        print("Edge not found.")
        return

    screen_width, screen_height = pyautogui.size()
    create_overlay_window(screen_width, screen_height)

    with FaceLandmarker.create_from_options(options) as landmarker:
        while not pr.window_should_close():
            screen_np, capture_width, capture_height = capture_window(window_id)
            frame_copy = screen_np.copy()
            frame_copy = cv2.blur(frame_copy, (27, 27))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=screen_np)
            face_landmarks_result = landmarker.detect(mp_image)
            face_landmarks_list = face_landmarks_result.face_landmarks

            scale_x = screen_width / capture_width
            scale_y = screen_height / capture_height

            pr.begin_drawing()
            pr.clear_background(pr.BLANK)

            for face in face_landmarks_list:
                # convexhull = cv2.convexHull(np.array(facial_landmarks))
                # mask = np.zeros((height, width), np.uint8)
                # cv2.fillConvexPoly(mask, convexhull, (255, 0, 0))
                # face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
                # for pt in zip(mask.nonzero()[0], mask.nonzero()[1]):
                #     [r, g, b] = face_extracted[pt[1], pt[0]]
                #     pr.draw_circle(int(pt[1]), int(pt[0]), 1, pr.Color(r, g, b, 255))
                for landmarks in face:
                    x, y = landmarks.x, landmarks.y
                    normed = normalized_to_pixel_coordinates(
                        x, y, screen_width, screen_height
                    )
                    if normed is not None:
                        x, y = normed
                        pr.draw_circle(int(x * scale_x), int(y * scale_y), 2, pr.WHITE)

            pr.end_drawing()
        pr.close_window()


if __name__ == "__main__":
    main()
