import cv2
from PIL import Image
from util import get_limits

# Define yellow color in BGR format (Blue: 0, Green: 255, Red: 255)
yellow = (0, 255, 255)

# OPTIMIZATION: Get lower and upper HSV limits for yellow ONCE outside the loop
lowerLimit, upperLimit = get_limits(color=yellow)

# Initialize webcam stream (try index 0, 1, or 2 depending on your camera setup)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # FIX: Safety check to ensure the frame was successfully captured
    if not ret:
        print("Failed to grab frame. Retrying...")
        continue

    # Convert frame from BGR to HSV color space
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Generate binary mask for yellow pixels
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # Convert mask to Pillow image format and extract bounding box coordinates
    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()

    # Draw green bounding box if a yellow object is detected
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Show live frame
    cv2.imshow('frame', frame)

    # Press 'q' to stop script and exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam hardware and close windows
cap.release()
cv2.destroyAllWindows()
