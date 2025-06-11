import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse

def load_point_cloud(bin_path):
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
    return pts

def build_colored_pointcloud(pts):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts[:, :3])
    zs = pts[:, 2]
    zs_normalized = (zs - zs.min()) / (zs.ptp() + 1e-8)
    colors = plt.get_cmap('cividis')(zs_normalized)[:, :3]
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc

def load_labels(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            h, w, l = map(float, parts[8:11])
            x, y, z = map(float, parts[11:14])
            label = parts[0]
            boxes.append({
                'center': np.array([x, y, z]),
                'size': (l, w, h),
                'label': label
            })
    return boxes

def create_box(center, size, color):
    l, w, h = size
    half_size = np.array([l/2, w/2, h/2])
    aabb = o3d.geometry.AxisAlignedBoundingBox(center - half_size, center + half_size)
    box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
    box.paint_uniform_color(color)
    return box

def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud with optional Z-projection for bounding boxes.")
    parser.add_argument('--project_z', action='store_true', help="Project box base to lowest Z inside XY footprint with margin")
    args = parser.parse_args()

    bin_path = "fskitti2/points/0000000.bin"
    label_path = "fskitti2/labels/0000000.txt"

    pts = load_point_cloud(bin_path)

    if args.project_z:
        lidar_z_offset = -0.165
        pts[:, 2] += lidar_z_offset  # only apply if projecting
        print(f"[INFO] Applied lidar_z_offset of {lidar_z_offset:.3f}")
    else:
        print("[INFO] Using labels as-is, without modifying Z or point cloud")
    pcd = build_colored_pointcloud(pts)


    boxes = load_labels(label_path)
    color_map = {
        'blue_cone':   (0, 0, 1),
        'yellow_cone': (1, 1, 0),
        'orange_cone': (1, 0.5, 0),
        'large_orange_cone': (0.5, 0, 0.5),
    }

    geoms = [pcd]
    for box in boxes:
        color = color_map.get(box['label'], (0.5, 0.5, 0.5))
        center = box['center'].copy()
        size = box['size']
        l, w, h = size
        margin = 0.05

        if args.project_z:
            min_x = center[0] - l / 2
            max_x = center[0] + l / 2
            min_y = center[1] - w / 2
            max_y = center[1] + w / 2

            mask = (
                (pts[:, 0] >= min_x) & (pts[:, 0] <= max_x) &
                (pts[:, 1] >= min_y) & (pts[:, 1] <= max_y)
            )
            points_in_box = pts[mask]

            if len(points_in_box) > 0:
                min_z = np.min(points_in_box[:, 2])
                adjusted_bottom_z = min_z - h * margin
                center[2] = adjusted_bottom_z + h / 2
                print(f"[{box['label']}] Adjusted base Z = {adjusted_bottom_z:.3f}")

        geoms.append(create_box(center, size, color))

    vis = o3d.visualization.Visualizer()
    vis.create_window("PointCloud with Fixed Z-Offset", width=900, height=600)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
