import cv2
import numpy as np

def get_limits(color):
    # 1. Convert BGR to HSV
    c = np.uint8([[color]])  # BGR color input
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    
    # 2. Extract the exact Hue integer value from the 3D array
    hue = hsvC[0][0][0]
    
    # 3. Calculate lower and upper threshold limits for Hue safely
    # Handling OpenCV's 0-179 Hue range limits prevents underflow/overflow
    lower_hue = max(0, hue - 10)
    upper_hue = min(179, hue + 10)
    
    # 4. Create the final 1D array limits for cv2.inRange()
    lowerLimit = np.array([lower_hue, 100, 100], dtype=np.uint8)
    upperLimit = np.array([upper_hue, 255, 255], dtype=np.uint8)
    
    return lowerLimit, upperLimit
