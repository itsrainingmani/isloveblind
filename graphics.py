import math
import cv2
import pyray as pr
from windows import *
from scipy.spatial import ConvexHull

# everything in this module should accept
# a) an array of predictions
# b) an np array of the current screen

# (x1, y1) - top left corner, (x2, y2) - bottom right corner


def draw_circle_on_face(preds, screen_img_np, color=pr.WHITE):
    for pred in preds:
        pred = pred if pred is not None else []
        [x1, y1, x2, y2, conf] = pred
        x1, y1, x2, y2 = (
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        )
        w = x2 - x1 + 5
        h = y2 - y1 + 5
        center_of_rect = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        if conf > 0.5:
            pr.draw_circle(
                center_of_rect[0],
                center_of_rect[1],
                math.sqrt(w**2 + h**2) / 2,
                color,
            )


def draw_ellipse_on_face(preds, screen_img_np, color=pr.WHITE):
    for pred in preds:
        pred = pred if pred is not None else []
        [x1, y1, x2, y2, conf] = pred
        x1, y1, x2, y2 = (
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        )
        w = x2 - x1 + 5
        h = y2 - y1 + 5
        [center_x, center_y] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        radius = math.sqrt(w**2 + h**2) / 2
        if conf > 0.5:
            # pr.draw_circle(
            #     center_x,
            #     center_y,
            #     radius,
            #     color,
            # )
            # the vertical should be roughly 15% larger than the horizontal axis
            pr.draw_ellipse(center_x, center_y, radius, radius * 1.15, color)


def draw_blur_on_face(preds, screen_img_np):
    for pred in preds:
        pred = pred if pred is not None else []
        # (x1, y1) - top left corner, (x2, y2) - bottom right corner
        [x1, y1, x2, y2, conf] = pred
        x1, y1, x2, y2 = (
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        )
        w = x2 - x1 + 5
        h = y2 - y1 + 5
        roi = screen_img_np[y1 : y1 + h, x1 : x1 + w]
        blur = cv2.GaussianBlur(roi, (41, 41), 0)
        for j in range(blur.shape[0]):
            for i in range(blur.shape[1]):
                [r, g, b] = blur[j, i]
                pr.draw_circle(
                    i + x1,
                    j + y1,
                    1,
                    pr.Color(r, g, b, 255),
                )


def draw_landmarks(preds, scale_x, scale_y, color=pr.WHITE):
    # Draw face Landmarks
    for pred in preds:
        pred = pred if pred is not None else []
        contours = [
            (int(landmark[0] * scale_x), int(landmark[1] * scale_y))
            for landmark in pred
        ]
        for c in contours:
            pr.draw_circle(int(c[0] * scale_x), int(c[1] * scale_y), 3, color)


def blank_within_contour(preds, screen_img_np, scale_x, scale_y, color=pr.WHITE):
    mask = np.zeros(screen_img_np.shape[:2], dtype=np.uint8)
    # Draw face Landmarks
    for pred in preds:
        pred = pred if pred is not None else []
        contours = [
            (int(landmark[0] * scale_x), int(landmark[1] * scale_y))
            for landmark in pred
        ]
        # Generate a Mask that covers the contour of the face using convex hull
        hull = ConvexHull(contours)
        vertex_pts = [contours[vertex] for vertex in hull.vertices]
        cv2.fillConvexPoly(mask, np.array(vertex_pts), color=(255, 0, 0))
        # blurred_face = cv2.bitwise_and(screen_blur, screen_blur, mask=mask)
        zipapped = zip(*np.nonzero(mask))
        for pt in zipapped:
            # [r, g, b] = blurred_face[pt[0], pt[1]][:]
            # r, g, b = 255, 0, 0
            pr.draw_circle(int(pt[1] * scale_x), int(pt[0] * scale_y), 2, pr.WHITE)


# pr.draw_rectangle(x1, y1, w, h, pr.WHITE)
# mask = np.zeros(screen_img_np.shape[:2], dtype=np.uint8)
# pr.draw_rectangle(x1, y1, x2 - x1, y2 - y1, pr.WHITE)
# cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 0, 0), -1)
# blurred_face = cv2.bitwise_and(screen_blur, screen_blur, mask=mask)
# zipapped = zip(*np.nonzero(mask))
# for pt in zipapped:
# [r, g, b] = blurred_face[pt[0], pt[1]][:]
# pr.draw_circle(
#     int(pt[1] * scale_x),
#     int(pt[0] * scale_y),
#     1,
#     # pr.Color(r, g, b, 255),
# )
# r, g, b = 255, 0, 0
# pr.draw_circle(int(pt[1] * scale_x), int(pt[0] * scale_y), 2, pr.WHITE)
