import cv2
import numpy as np
from collections import deque, Counter

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controls", 400, 180)
cv2.moveWindow("Controls", 40, 40)
cv2.createTrackbar("Min Area", "Controls", 800, 10000, nothing)
cv2.createTrackbar("Edge Sensitivity", "Controls", 55, 200, nothing)
cv2.createTrackbar("Approx Precision", "Controls", 30, 60, nothing)

cv2.namedWindow("High-Precision Shape Detector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("High-Precision Shape Detector", 960, 540)
cv2.moveWindow("High-Precision Shape Detector", 460, 40)

controls_bg = np.zeros((1, 400, 3), dtype=np.uint8)
tracked_objects = {}
next_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    min_area = cv2.getTrackbarPos("Min Area", "Controls")
    thresh_val = cv2.getTrackbarPos("Edge Sensitivity", "Controls")
    eps_factor = cv2.getTrackbarPos("Approx Precision", "Controls") / 1000.0

    smooth = cv2.bilateralFilter(frame, 5, 50, 50)

    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    edges_h = cv2.Canny(h, thresh_val, thresh_val * 2)
    edges_s = cv2.Canny(s, thresh_val, thresh_val * 2)
    edges_v = cv2.Canny(v, thresh_val, thresh_val * 2)

    edges = cv2.bitwise_or(edges_h, edges_s)
    edges = cv2.bitwise_or(edges, edges_v)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    current_detections = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > 300000:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = float(area) / hull_area
        if solidity < 0.85:
            continue

        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
        vertices = len(approx)

        circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0

        shape = None
        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            x, y, w, h_box = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h_box
            shape = "Square" if 0.88 <= aspect_ratio <= 1.12 else "Rectangle"
        elif vertices == 5:
            shape = "Pentagon"
        elif vertices == 6:
            shape = "Hexagon"
        elif vertices > 6 and circularity > 0.76:
            shape = "Circle"

        if shape:
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                is_duplicate = False
                for _, (existing_cx, existing_cy), _ in current_detections:
                    if np.hypot(cx - existing_cx, cy - existing_cy) < 30:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    current_detections.append((approx, (cx, cy), shape))

    updated_objects = {}
    used_ids = set()

    for approx, (cx, cy), shape in current_detections:
        best_id = None
        min_dist = 70

        for obj_id, data in tracked_objects.items():
            if obj_id in used_ids:
                continue
            prev_cx, prev_cy = data["center"]
            dist = np.hypot(cx - prev_cx, cy - prev_cy)
            if dist < min_dist:
                min_dist = dist
                best_id = obj_id

        if best_id is None:
            best_id = next_id
            next_id += 1
            history = deque(maxlen=25)
            locked_shape = None
        else:
            history = tracked_objects[best_id]["history"]
            locked_shape = tracked_objects[best_id]["locked_shape"]

        history.append(shape)

        if locked_shape is None and len(history) >= 12:
            top_shape, count = Counter(history).most_common(1)[0]
            if count >= 10:
                locked_shape = top_shape

        used_ids.add(best_id)
        updated_objects[best_id] = {
            "center": (cx, cy),
            "approx": approx,
            "history": history,
            "locked_shape": locked_shape,
            "missed": 0
        }

    for obj_id, data in tracked_objects.items():
        if obj_id not in used_ids and data["missed"] < 8:
            data["missed"] += 1
            updated_objects[obj_id] = data

    tracked_objects = updated_objects

    for obj_id, data in tracked_objects.items():
        if data["locked_shape"] is not None:
            approx = data["approx"]
            cx, cy = data["center"]
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            cv2.putText(frame, data["locked_shape"], (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Controls", controls_bg)
    cv2.imshow("High-Precision Shape Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
