import cv2
import numpy as np

image = np.ones((600, 800, 3), dtype=np.uint8) * 255

triangle_pts = np.array([[150, 50], [50, 200], [250, 200]], np.int32)
cv2.fillPoly(image, [triangle_pts], (200, 100, 100))

cv2.rectangle(image, (350, 60), (520, 190), (100, 200, 100), -1)

pentagon_pts = np.array([[670, 50], [750, 110], [720, 200], [620, 200], [590, 110]], np.int32)
cv2.fillPoly(image, [pentagon_pts], (100, 100, 200))

hexagon_pts = np.array([[150, 320], [230, 360], [230, 440], [150, 480], [70, 440], [70, 360]], np.int32)
cv2.fillPoly(image, [hexagon_pts], (200, 200, 100))

cv2.circle(image, (450, 400), 80, (200, 100, 200), -1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

for i, contour in enumerate(contours):
    if i == 0:
        continue

    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    cv2.drawContours(image, [approx], 0, (0, 0, 0), 4)

    x, y, width, height = cv2.boundingRect(approx)
    x_mid = int(x + width / 3)
    y_mid = int(y + height / 1.5)
    coords = (x_mid, y_mid)

    color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_DUPLEX

    if len(approx) == 3:
        cv2.putText(image, "Triangle", coords, font, 0.7, color, 2)
    elif len(approx) == 4:
        cv2.putText(image, "Rectangle", coords, font, 0.7, color, 2)
    elif len(approx) == 5:
        cv2.putText(image, "Pentagon", coords, font, 0.7, color, 2)
    elif len(approx) == 6:
        cv2.putText(image, "Hexagon", coords, font, 0.7, color, 2)
    else:
        cv2.putText(image, "Circle", coords, font, 0.7, color, 2)

cv2.imshow("Detected Shapes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
