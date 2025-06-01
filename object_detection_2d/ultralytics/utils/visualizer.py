import cv2
from typing import List, Union

def get_random_color(seed):
    # Define colors in BGR format to match OpenCV's default color handling
    color_list = [
        [168, 86, 51],    # #3356A8 -> BGR
        [41, 88, 217],    # #D95829 -> BGR
        [40, 67, 204],    # #CC4328 -> BGR
        [154, 154, 150],  # #969A9A -> BGR
        [37, 172, 217]    # #D9AC25 -> BGR
    ]
    # Select color based on class ID (seeded for reproducibility)
    return color_list[seed % len(color_list)]

def draw_detections(img, bboxes, classes, class_labels, confidences, show_boxes=True, line_width=None):
    # Draw bounding boxes and labels on the image with confidence scores
    for bbox, cls, conf in zip(bboxes, classes, confidences):
        x1, y1, x2, y2 = bbox
        color = get_random_color(int(cls))
        
        # Draw bounding box if enabled
        if show_boxes:
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width if line_width else max(1, min(img.shape[:2]) // 300))

        # Display label with confidence
        if class_labels:
            label = f"{class_labels[int(cls)]} {conf:.3f}"
            img = cv2.putText(img, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, color, 1, cv2.LINE_AA)
    return img
