import math
import cv2
import pyray as pr
from windows import *
from scipy.spatial import ConvexHull

# everything in this module should accept
# a) an array of predictions
# b) an np array of the current screen

# (x1, y1) - top left corner, (x2, y2) - bottom right corner


def draw_circle_on_face(preds, color=pr.WHITE):
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
        if conf > 0.7:
            pr.draw_circle(
                center_of_rect[0],
                center_of_rect[1],
                math.sqrt(w**2 + h**2) / 2,
                color,
            )


def draw_ellipse_on_face(preds, color=pr.WHITE):
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
        if conf > 0.7:
            # the vertical should be roughly 15% larger than the horizontal axis
            pr.draw_ellipse(center_x, center_y, radius * 0.95, radius * 1.16, color)


def draw_average_color_on_face(preds, screen_img_np):
    for pred in preds:
        pred = pred if pred is not None else []
        [x1, y1, x2, y2, conf] = pred
        x1, y1, x2, y2 = (
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        )
        if conf > 0.7:
            # Define the center and axes of the ellipse
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            w = x2 - x1 + 5
            h = y2 - y1 + 5
            radius = math.sqrt(w**2 + h**2) / 2
            axes_length = int(radius * 0.95), int(radius * 1.16)

            # Create a mask for the ellipse
            mask = np.zeros(screen_img_np.shape[:2], dtype=np.uint8)
            cv2.ellipse(
                mask, (center_x, center_y), axes_length, 0, 0, 360, (255, 0, 0), -1
            )

            # Extract the region of interest
            roi = cv2.bitwise_and(screen_img_np, screen_img_np, mask=mask)

            # Calculate the average color within the ellipse
            total_pixels = np.count_nonzero(mask)
            if total_pixels > 0:
                sum_color = cv2.sumElems(roi)
                avg_color = (
                    int(sum_color[0] / total_pixels),
                    int(sum_color[1] / total_pixels),
                    int(sum_color[2] / total_pixels),
                )
            else:
                avg_color = (0, 0, 0)

            # Draw the ellipse with the average color
            pr.draw_ellipse(
                center_x,
                center_y,
                axes_length[0],
                axes_length[1],
                pr.Color(*avg_color, 255),
            )


def draw_squiggle_effect_on_face(preds, screen_img_np, scale_x, scale_y):
    mask = np.zeros(screen_img_np.shape[:2], dtype=np.uint8)
    for pred in preds:
        pred = pred if pred is not None else []
        [x1, y1, x2, y2, conf] = pred
        x1, y1, x2, y2 = (
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        )
        if conf > 0.7:
            # Define the center and axes of the ellipse
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            w = x2 - x1 + 5
            h = y2 - y1 + 5
            radius = math.sqrt(w**2 + h**2) / 2
            axes_length = int(radius * 0.95), int(radius * 1.16)

            # Create a mask for the ellipse
            cv2.ellipse(
                mask, (center_x, center_y), axes_length, 0, 0, 360, (255, 0, 0), -1
            )

            # Extract the region of interest
            roi = cv2.bitwise_and(screen_img_np, screen_img_np, mask=mask)

            # Apply squiggle effect
            squiggle_roi = np.zeros_like(roi)
            for y in range(roi.shape[0]):
                for x in range(roi.shape[1]):
                    if mask[y, x] == 255:
                        # Apply a squiggle transformation
                        offset_x = int(5 * math.sin(2 * math.pi * y / 20))
                        offset_y = int(5 * math.cos(2 * math.pi * x / 20))
                        new_x = min(max(x + offset_x, 0), roi.shape[1] - 1)
                        new_y = min(max(y + offset_y, 0), roi.shape[0] - 1)
                        squiggle_roi[y, x] = roi[new_y, new_x]

            # Draw only the pixels within the squiggle_roi
            for y in range(squiggle_roi.shape[0]):
                for x in range(squiggle_roi.shape[1]):
                    if mask[y, x] == 255:  # Only draw within the ellipse
                        [r, g, b] = squiggle_roi[y, x]
                        if (
                            r != 0 or g != 0 or b != 0
                        ):  # Check if the pixel is part of the squiggle effect
                            pr.draw_circle(
                                int(x * scale_x),
                                int(y * scale_y),
                                1,
                                pr.Color(r, g, b, 255),
                            )


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
        w = x2 - x1
        h = y2 - y1
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


def avg_within_contour(preds, screen_img_np, scale_x, scale_y, color=pr.WHITE):
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
        roi = cv2.bitwise_and(screen_img_np, screen_img_np, mask=mask)
        # Apply squiggle effect
        squiggle_roi = np.zeros_like(roi)
        for y in range(roi.shape[0]):
            for x in range(roi.shape[1]):
                if mask[y, x] == 255:
                    # Apply a squiggle transformation
                    offset_x = int(5 * math.sin(2 * math.pi * y / 20))
                    offset_y = int(5 * math.cos(2 * math.pi * x / 20))
                    new_x = min(max(x + offset_x, 0), roi.shape[1] - 1)
                    new_y = min(max(y + offset_y, 0), roi.shape[0] - 1)
                    squiggle_roi[y, x] = roi[new_y, new_x]
        # blurred_face = cv2.bitwise_and(screen_blur, screen_blur, mask=mask)
        zipapped = zip(*np.nonzero(mask))
        for pt in zipapped:
            # [r, g, b] = blurred_face[pt[0], pt[1]][:]
            # r, g, b = 255, 0, 0
            pr.draw_circle(int(pt[1] * scale_x), int(pt[0] * scale_y), 2, pr.WHITE)
