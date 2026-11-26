import cv2
import numpy as np


def get_limits(color):
    c = np.uint8(((color,),))
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = int(hsvC.item(0, 0, 0))

    # Handle red wrap-around at 0 / 179
    if hue >= 170:
        return [
            (
                np.array((hue - 10, 120, 70), dtype=np.uint8),
                np.array((179, 255, 255), dtype=np.uint8),
            ),
            (
                np.array((0, 120, 70), dtype=np.uint8),
                np.array((hue - 170, 255, 255), dtype=np.uint8),
            ),
        ]
    elif hue <= 10:
        return [
            (
                np.array((170 + hue, 120, 70), dtype=np.uint8),
                np.array((179, 255, 255), dtype=np.uint8),
            ),
            (
                np.array((0, 120, 70), dtype=np.uint8),
                np.array((hue + 10, 255, 255), dtype=np.uint8),
            ),
        ]
    else:
        lowerLimit = np.array((hue - 10, 120, 70), dtype=np.uint8)
        upperLimit = np.array((hue + 10, 255, 255), dtype=np.uint8)
        return [(lowerLimit, upperLimit)]
