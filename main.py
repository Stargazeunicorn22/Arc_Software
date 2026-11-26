import cv2
import numpy as np
from util import get_limits

colors = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255),
}

color_limits = {name: get_limits(bgr) for name, bgr in colors.items()}
kernel = np.ones((5, 5), np.uint8)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam. Try changing VideoCapture(0) to 1.")

# Tracks what was detected in the previous frame to prevent console spam
last_detected_colors = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Retrying...")
        continue

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsvImage = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    black_screen = np.zeros_like(frame)

    detected_colors = set()

    for name, limits in color_limits.items():
        mask = None
        for lower, upper in limits:
            current_mask = cv2.inRange(hsvImage, lower, upper)
            mask = (
                current_mask
                if mask is None
                else cv2.bitwise_or(mask, current_mask)
            )

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        bgr_color = colors[name]
        black_screen[mask > 0] = bgr_color

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            if cv2.contourArea(cnt) > 800:
                detected_colors.add(name)
                cv2.drawContours(frame, [cnt], -1, bgr_color, 3)
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.putText(
                    frame,
                    name,
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    bgr_color,
                    2,
                )

    # 1. Print to console ONLY when the detected colors change (not constantly)
    if detected_colors != last_detected_colors:
        for color in detected_colors - last_detected_colors:
            print(f"OK {color} detected")
        last_detected_colors = detected_colors.copy()

    # 2. Show "OK <Color> detected" on the Black Screen window
    y_offset = 40
    for color in sorted(detected_colors):
        cv2.putText(
            black_screen,
            f"OK {color} detected",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            colors[color],
            2,
        )
        y_offset += 40

    cv2.imshow("Live Detection", frame)
    cv2.imshow("Black Screen Color View", black_screen)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
