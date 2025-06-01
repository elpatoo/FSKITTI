import cv2
import torch
from ultralytics import YOLO

class YoloWrapper:
    def __init__(self, weights, conf_thresh, iou_thresh, input_size, device, half=False, show_features=False):
        self.model = YOLO(weights)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.input_size = input_size
        self.device = device
        self.half = half
        self.show_features = show_features

        if self.half and self.device == 'cuda':
            self.model.half()

    def inference(self, img):
        img_resized = cv2.resize(img, self.input_size)
        results = self.model(img_resized, imgsz=self.input_size, visualize=self.show_features)

        detections = []
        for det in results[0].boxes:
            if hasattr(det, 'data') and det.data.size(1) == 6:
                x1, y1, x2, y2, conf, cls = det.data[0].tolist()
                if conf >= self.conf_thresh:
                    detections.append([x1, y1, x2, y2, conf, cls])

        return torch.tensor(detections, dtype=torch.float32) if detections else torch.empty((0, 6))
