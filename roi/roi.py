import cv2
from morphological import detect_forehead_roi
from viola_jones import detect_face_regions

# Example usage
# output_img, roi = detect_forehead_roi("../data/images/1569918345_lica-1.jpg")
#
# if roi is not None:
#     cv2.imshow("Forehead ROI", output_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("No face detected in the image")

# Example usage
output_img, forehead, cheeks = detect_face_regions("../data/images/1569918345_lica-1.jpg")

if forehead is not None and cheeks is not None:
    cv2.imshow("Face Regions", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No face detected in the image")