import cv2
import numpy as np


def detect_face_regions(image_path):
    """
    Detects face regions (forehead and cheeks) using Haar Cascade classifier
    and returns the image with marked ROIs.

    Args:
        image_path (str): Path to the input image file

    Returns:
        numpy.ndarray: Image with forehead and cheeks ROIs drawn
        tuple: Coordinates of forehead ROI (x, y, w, h)
        tuple: Coordinates of cheeks ROI (x, y, w, h)
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image from path: {}".format(image_path))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return image, None, None

    # Get the first face (assuming single face in image)
    x, y, w, h = faces[0]

    # Define regions of interest
    forehead_roi = (x, y, w, h // 3)  # Top 1/3 of face
    cheeks_roi = (x + w // 4, y + h // 3, w // 2, h // 3)  # Middle 1/3, centered

    # Draw rectangles on original image
    output_image = image.copy()

    # Draw forehead (green rectangle)
    cv2.rectangle(output_image,
                  (forehead_roi[0], forehead_roi[1]),
                  (forehead_roi[0] + forehead_roi[2], forehead_roi[1] + forehead_roi[3]),
                  (0, 255, 0), 2)

    # Draw cheeks (blue rectangle)
    cv2.rectangle(output_image,
                  (cheeks_roi[0], cheeks_roi[1]),
                  (cheeks_roi[0] + cheeks_roi[2], cheeks_roi[1] + cheeks_roi[3]),
                  (255, 0, 0), 2)

    return output_image, forehead_roi, cheeks_roi