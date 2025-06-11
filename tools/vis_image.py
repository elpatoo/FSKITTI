import os
import cv2
import matplotlib.pyplot as plt

def load_kitti_labels(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            label = parts[0]
            xmin, ymin, xmax, ymax = map(float, parts[4:8])
            boxes.append({
                'label': label,
                'bbox': (xmin, ymin, xmax, ymax)
            })
    return boxes

def show_image_with_boxes(image_path, label_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load boxes
    boxes = load_kitti_labels(label_path)

    # Plot image with bounding boxes
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    for box in boxes:
        xmin, ymin, xmax, ymax = box['bbox']
        width = xmax - xmin
        height = ymax - ymin
        rect = plt.Rectangle((xmin, ymin), width, height,
                             linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, box['label'], color='red',
                fontsize=10, backgroundcolor='white')

    plt.title("2D Bounding Boxes from KITTI Labels")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    image_path = "fskitti/camera_central_noise_rain/images/0000000.png"
    label_path = "fskitti/camera_central_noise_rain/labels/0000000.txt"
    show_image_with_boxes(image_path, label_path)

if __name__ == "__main__":
    main()
