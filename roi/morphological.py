import cv2
import numpy as np


def detect_forehead_roi(image_path):
    """
    Detects forehead region on a face image and returns the image with marked ROI.

    Args:
        image_path (str): Path to the input image file

    Returns:
        numpy.ndarray: Image with forehead ROI rectangle drawn
        tuple: Coordinates of forehead ROI (x, y, w, h)
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image from path: {}".format(image_path))

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create skin color mask
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([30, 170, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image, None

    # Get largest contour (assuming it's the face)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Define forehead region (top 1/3 of face)
    forehead_roi = (x, y, w, h // 3)

    # Draw rectangle on original image
    output_image = image.copy()
    cv2.rectangle(output_image,
                  (forehead_roi[0], forehead_roi[1]),
                  (forehead_roi[0] + forehead_roi[2], forehead_roi[1] + forehead_roi[3]),
                  (0, 255, 0), 2)  # Green rectangle

    return output_image, forehead_roi