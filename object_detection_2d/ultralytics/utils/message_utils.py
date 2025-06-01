import cv2
from common_msgs.msg import Yolo, YoloPoint
from .visualizer import draw_detections

def prepare_yolo_message(detections, header, orig_w, orig_h, img_size):
    """
    Prepare a Yolo message by scaling detections to the original image dimensions.
    Each detection is populated in a custom YoloPoint message.
    """
    yolo_msg = Yolo()
    yolo_msg.header = header

    # Calculate scaling factors to convert back to original image dimensions
    scale_x, scale_y = orig_w / img_size, orig_h / img_size
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

        # Populate YoloPoint message for each detection
        yolo_point = YoloPoint(
            x=int(x1 + (x2 - x1) / 2),
            y=int(y1 + (y2 - y1) / 2),
            size_x=int(x2 - x1),
            size_y=int(y2 - y1),
            results_id=int(cls),
            results_score=conf
        )
        yolo_msg.detections.append(yolo_point)

    return yolo_msg

def draw_visualization(img, detections, orig_w, orig_h, img_size, class_labels, show_boxes=True, line_width=None):
    """
    Draw visualizations of detections on the image. Bounding boxes are scaled to original dimensions.
    """
    scaled_bboxes = [
        [int(x1 * orig_w / img_size), int(y1 * orig_h / img_size), int(x2 * orig_w / img_size), int(y2 * orig_h / img_size)]
        for x1, y1, x2, y2, _, _ in detections.tolist()
    ]
    classes = [int(cls) for _, _, _, _, _, cls in detections.tolist()]
    confidences = [float(conf) for _, _, _, _, conf, _ in detections.tolist()]

    # Use the draw_detections function from visualizer.py to handle box drawing and labeling
    return draw_detections(img, scaled_bboxes, classes, class_labels, confidences, show_boxes=show_boxes, line_width=line_width)
