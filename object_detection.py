import numpy as np
import cv2
import math


def angle_cos(p_, q_, r_):
    v1_ = p_ - q_
    v2_ = r_ - q_
    return abs(np.dot(v1_, v2_) / (np.linalg.norm(v1_) * np.linalg.norm(v2_)))


def circle_center_from_3pts(A_, B_, C_):
    BA = B_ - A_
    CA = C_ - A_
    det = BA[0] * CA[1] - BA[1] * CA[0]
    if abs(det) < 1e-6:
        return None
    d = 2 * det
    a2, b2, c2 = np.dot(A_, A_), np.dot(B_, B_), np.dot(C_, C_)
    ux = (CA[1] * (b2 - a2) - BA[1] * (c2 - a2)) / d
    uy = (BA[0] * (c2 - a2) - CA[0] * (b2 - a2)) / d
    return np.array([ux, uy], dtype=np.float32)


def mean_angle_deg(angles_):
    s = sum(math.sin(math.radians(a)) for a in angles_)
    c = sum(math.cos(math.radians(a)) for a in angles_)
    return math.degrees(math.atan2(s, c))


def process_image_fast(img_):
    hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)

    LOWER_YELLOW = np.array([23, 80, 70])
    UPPER_YELLOW = np.array([40, 255, 255])
    LOWER_BLUE = np.array([100, 120, 50])
    UPPER_BLUE = np.array([130, 255, 255])
    YELLOW_AREA_MIN = 1200
    BLUE_AREA_MIN = 3
    BLUE_AREA_MAX = 400

    # ---------- 1. 黄色零件 ----------
    mask_y = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cv2.dilate(mask_y, kernel, iterations=2, dst=mask_y)

    # cv2.imshow("mask_yellow", mask_y)

    contours, _ = cv2.findContours(mask_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []

    for cnt in contours:
        if cv2.contourArea(cnt) < YELLOW_AREA_MIN:
            continue

        (cx, cy), r = cv2.minEnclosingCircle(cnt)

        # ---------- 2. ROI ----------
        x0 = max(int(cx - r), 0)
        y0 = max(int(cy - r), 0)
        x1 = min(int(cx + r), img_.shape[1])
        y1 = min(int(cy + r), img_.shape[0])

        roi = img_[y0:y1, x0:x1]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask_b = cv2.inRange(hsv_roi, LOWER_BLUE, UPPER_BLUE)
        mask_b = cv2.morphologyEx(
            mask_b,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        cv2.imshow("b_mask", mask_b)
        cnts_b, _ = cv2.findContours(mask_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pts = []
        for c in cnts_b:
            a = cv2.contourArea(c)
            if BLUE_AREA_MIN < a < BLUE_AREA_MAX:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    px = M["m10"] / M["m00"] + x0
                    py = M["m01"] / M["m00"] + y0
                    pts.append([px, py])
        # print(f"Detected {len(pts)} blue points in part at ({cx}, {cy})")
        if len(pts) != 3:
            continue

        # 不去畸变
        pts = np.array(pts, dtype=np.float32)

        A, B, C = pts

        # ---------- 3. 圆心 ----------
        center = circle_center_from_3pts(A, B, C)
        if center is None:
            continue

        # ---------- 4. 方向 ----------
        cos_vals = [
            angle_cos(B, A, C),
            angle_cos(A, B, C),
            angle_cos(A, C, B)
        ]
        idx = np.argmax(cos_vals)
        direction_pt = pts[idx]

        vec = direction_pt - center
        theta = math.degrees(math.atan2(vec[1], vec[0]))

        # 统一到 [-90, 90] 或 [-180, 180]
        if theta > 0:
            theta -= 180
        else:
            theta += 180

        results.append((center, theta))
    for result in results:
        cv2.circle(img_, (int(result[0][0]),int(result[0][1])), 5, (0, 0, 255), -1)
    return img_, results
