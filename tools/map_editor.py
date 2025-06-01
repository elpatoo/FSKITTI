#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import datetime as dt
import getpass as gt
import tkinter as tk
from tkinter import messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import List, Set, Tuple, Optional, Dict, Any, Generator
import open3d as o3d
import matplotlib.pyplot as plt
import rosbag
import rospy
import tf
from tf.transformations import (
    euler_from_quaternion,
    quaternion_slerp,
    quaternion_matrix
)
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Int16
from nav_msgs.msg import Odometry
from tf.msg import tfMessage
from geometry_msgs.msg import Transform
import cv2
from cv_bridge import CvBridge, CvBridgeError
import bisect

# YOLO Ultralytics imports
from object_detection_2d.ultralytics.utils.yolo_utils import YoloWrapper

# --- Topics ---
SLAM_MAP_TOPIC    = "/estimation/slam/landmarks"
CAR_STATE_TOPIC   = "/estimation/odom3D"
POINTCLOUD_TOPIC  = "/hesai/pandar"
IMAGE_TOPIC       = "/arena_camera_node/image_raw"
LAP_COUNT_TOPIC   = "/common/mission_tracker/lap_counter"

LOCAL_FRAME = "pandar40p"
CONE_FRAME  = "map"

# Cone color enums
UNKNOWN_C = 0
YELLOW_C = 1
BLUE_C   = 2
BIG_C    = 3
ORANGE_C = 4

# KITTI dims & color mapping
DIMENSIONS = {
    YELLOW_C: (0.228,0.228,0.325),
    BLUE_C:   (0.228,0.228,0.325),
    ORANGE_C:(0.228,0.228,0.325),
    BIG_C:    (0.285,0.285,0.505),
}

LABEL_NAME = {
    YELLOW_C: 'yellow_cone',
    BLUE_C:   'blue_cone',
    ORANGE_C: 'orange_cone',
    BIG_C:    'large_orange_cone',
}

VIS_COLORS = {
    'yellow_cone': (1,1,0),
    'blue_cone':   (0,0,1),
    'orange_cone': (1,0.5,0),
    'large_orange_cone':    (0.5,0,0.5),
}

def parse_classes_file(path: str) -> List[str]:
    try:
        with open(path, "r") as f:
            return [l.strip() for l in f if l.strip()]
    except Exception as e:
        rospy.logwarn(f"[SceneGenerator] Failed to load classes file '{path}': {e}")
        return []

def normalize_frame_name(frame: str) -> str:
    return frame.strip("/")

class SceneGenerator:
    def __init__(
        self,
        selected_file: str,
        label_every: int = 10,
        ego_motion_compensate: bool = True,
        lidar_z_offset: float = 1.15,
        visualize: bool = False,
        lidar_freq: float = 20.0,
        min_range: float = 2.0,
        max_range: float = 20.0,
        expansion_coeff: float = 0.0,
        centroid_correction: bool = False,
        cluster_radius: float = 0.5,
        min_cluster_points: int = 5,
        min_angle: float = -70.0,
        max_angle: float = 70.0,
        # YOLO args
        yolo_weights: str = "",
        yolo_classes: str = "",
        yolo_img_size: int = 640,
        yolo_conf_thresh: float = 0.35,
        yolo_iou_thresh: float = 0.45,
        yolo_device: str = "cpu",
        yolo_half: bool = False,
        max_img_lag: float = 0.1,
    ):
        self.selected_file = selected_file
        self.label_every = label_every
        self.ego_motion_compensate = ego_motion_compensate
        self.lidar_z_offset = lidar_z_offset
        self.visualize = visualize
        self.lidar_freq = lidar_freq
        self.min_range = min_range
        self.max_range = max_range
        self.expansion_coeff = expansion_coeff
        self.centroid_correction = centroid_correction
        self.cluster_radius = cluster_radius
        self.min_cluster_points = min_cluster_points
        self.min_angle = min_angle
        self.max_angle = max_angle
        # RGBA mapping index → (r,g,b,a)
        self.color_map = {
            UNKNOWN_C: (0.5, 0.5, 0.5, 0.8),  # gray for any unknown
            BLUE_C:    (0.0, 0.0, 1.0, 0.8),
            YELLOW_C:  (1.0, 1.0, 0.0, 0.8),
            ORANGE_C:  (1.0, 0.5, 0.0, 0.8),
            BIG_C:     (0.5, 0.0, 0.5, 0.8),
        }

        # YOLO settings
        self.yolo_weights = yolo_weights
        self.yolo_classes = yolo_classes
        self.yolo_img_size = yolo_img_size
        self.yolo_conf_thresh = yolo_conf_thresh
        self.yolo_iou_thresh = yolo_iou_thresh
        self.yolo_device = yolo_device
        self.yolo_half = yolo_half
        self.max_img_lag = max_img_lag

        # Load odometry and images upfront
        self.bridge = CvBridge()  # <-- Must come first

        # Load odometry and images upfront
        self.odom_buffer = self._load_odometry()
        self.image_buffer = self._load_images()
        self.tf_transforms: Dict[Tuple[str, str], Any] = {}


        # Initialize YOLO model if requested
        self.class_labels = []
        if self.yolo_weights:
            rospy.loginfo(f"[SceneGenerator] Loading YOLO model {self.yolo_weights} on {self.yolo_device}")
            self.yolo_model = YoloWrapper(
                weights=self.yolo_weights,
                conf_thresh=self.yolo_conf_thresh,
                iou_thresh=self.yolo_iou_thresh,
                input_size=(self.yolo_img_size, self.yolo_img_size),
                device=self.yolo_device,
                half=self.yolo_half,
                show_features=False
            )
            self.class_labels = parse_classes_file(self.yolo_classes)
            rospy.loginfo("[SceneGenerator] YOLO model loaded")
            print(f"[DEBUG] Class labels loaded: {self.class_labels}")

    def _load_odometry(self) -> List[Dict[str, Any]]:
        # ... unchanged from original ...
        odom_buf = []
        bag = rosbag.Bag(self.selected_file, 'r')
        for _, msg, _ in bag.read_messages(topics=[CAR_STATE_TOPIC]):
            if isinstance(msg, Odometry):
                t = msg.header.stamp.to_sec()
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                odom_buf.append({
                    "t": t,
                    "trans": np.array([p.x, p.y, p.z], dtype=np.float64),
                    "quat": np.array([q.x, q.y, q.z, q.w], dtype=np.float64),
                })
        bag.close()
        odom_buf.sort(key=lambda e: e["t"])
        return odom_buf

    def _load_images(self) -> List[Dict[str, Any]]:
        return []

    def _get_closest_image(self, stamp) -> Optional[np.ndarray]:
        ts = stamp.to_sec()
        best_img = None
        best_lag = float("inf")
        try:
            with rosbag.Bag(self.selected_file, 'r') as bag:
                for _, msg, _ in bag.read_messages(topics=[IMAGE_TOPIC]):
                    t = msg.header.stamp.to_sec()
                    lag = abs(t - ts)
                    if lag < best_lag:
                        try:
                            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                            best_img = img
                            best_lag = lag
                        except Exception:
                            continue
                    if lag < self.max_img_lag:
                        break
        except Exception as e:
            print(f"[ERROR] Lazy image load failed: {e}")
            return None

        if best_lag <= self.max_img_lag:
            print(f"[DEBUG] Closest image lag: {best_lag:.3f}s (allowed: {self.max_img_lag:.3f}s)")
            return best_img
        print("[DEBUG] Image too far in time, skipping.")
        return None

    def _get_pose_matrix_at(self, query_t: float) -> np.ndarray:
        buf = self.odom_buffer
        if not buf:
            return np.eye(4)
        if query_t <= buf[0]["t"]:
            e = buf[0]
        elif query_t >= buf[-1]["t"]:
            e = buf[-1]
        else:
            lo, hi = 0, len(buf) - 1
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if buf[mid]["t"] < query_t:
                    lo = mid
                else:
                    hi = mid
            e0, e1 = buf[lo], buf[hi]
            t0, t1 = e0["t"], e1["t"]
            ratio = (query_t - t0) / (t1 - t0) if t1 > t0 else 0.0
            trans = (1 - ratio) * e0["trans"] + ratio * e1["trans"]
            quat = quaternion_slerp(e0["quat"], e1["quat"], ratio)
            M = quaternion_matrix(quat)
            M[:3, 3] = trans
            return M
        M = quaternion_matrix(e["quat"])
        M[:3, 3] = e["trans"]
        return M

    def _update_tf(self, tf_msg: tfMessage):
        for tr in tf_msg.transforms:
            key = (
                normalize_frame_name(tr.header.frame_id),
                normalize_frame_name(tr.child_frame_id),
            )
            self.tf_transforms[key] = tr

    def _transform_to_matrix(self, transform) -> np.ndarray:
        t = np.array([
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
        ])
        q = np.array([
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ])
        M = tf.transformations.quaternion_matrix(q)
        M[:3, 3] = t
        return M

    def _invert_transform(self, transform):
        M = self._transform_to_matrix(transform)
        invM = np.linalg.inv(M)
        inv_t = Transform()
        inv_t.translation.x, inv_t.translation.y, inv_t.translation.z = invM[:3, 3]
        q = tf.transformations.quaternion_from_matrix(invM)
        inv_t.rotation.x, inv_t.rotation.y, inv_t.rotation.z, inv_t.rotation.w = q
        return inv_t

    def _lookup_transform_chain(
        self,
        target_frame: str,
        source_frame: str,
        _time
    ) -> Optional[Any]:
        target, source = normalize_frame_name(target_frame), normalize_frame_name(source_frame)
        visited, stack = set(), [(source, np.eye(4))]
        while stack:
            cf, cm = stack.pop()
            if cf == target:
                tmsg = Transform()
                tmsg.translation.x, tmsg.translation.y, tmsg.translation.z = cm[:3, 3]
                q = tf.transformations.quaternion_from_matrix(cm)
                tmsg.rotation.x, tmsg.rotation.y, tmsg.rotation.z, tmsg.rotation.w = q
                from std_msgs.msg import Header
                class FakeTS:
                    def __init__(self):
                        self.header = Header()
                        self.child_frame_id = target
                        self.transform = tmsg
                return FakeTS()
            visited.add(cf)
            for (p, c), tr in self.tf_transforms.items():
                p_, c_ = normalize_frame_name(p), normalize_frame_name(c)
                if p_ == cf and c_ not in visited:
                    M = self._transform_to_matrix(tr.transform)
                    stack.append((c_, cm @ M))
                elif c_ == cf and p_ not in visited:
                    M = self._transform_to_matrix(self._invert_transform(tr.transform))
                    stack.append((p_, cm @ M))
        print(f"[TF Chain Error] {source} -> {target}")
        return None

    def _compensate_point_cloud(self, pc_array: np.ndarray, header_stamp) -> np.ndarray:
        # robustly detect timestamp units
        ts_raw = pc_array[:, 4].astype(np.float64)
        # if in microseconds (<1e8), convert to seconds; else nanoseconds
        if ts_raw.mean() > 1e7:
            rel_ts = ts_raw * 1e-9
        else:
            rel_ts = ts_raw * 1e-6
        sweep_end   = header_stamp.to_sec()
        sweep_start = sweep_end - 1.0 / self.lidar_freq
        T_end       = self._get_pose_matrix_at(sweep_end)

        out = np.zeros_like(pc_array)
        for i, p in enumerate(pc_array):
            x, y, z, inten, _ = p
            t_pt = sweep_start + rel_ts[i]
            T_pt = self._get_pose_matrix_at(t_pt)
            rel = np.linalg.inv(T_pt) @ T_end
            P   = np.array([x, y, z, 1.0])
            Pc  = rel @ P
            out[i, :3] = Pc[:3]
            out[i, 3]  = inten
            out[i, 4]  = p[4]
        return out

    def _get_local_cones(
        self,
        cones: List[Tuple[float, float, int]],
        transform,
        pc_array: np.ndarray
    ) -> Tuple[List[Tuple[float, float, int]], float]:
        ori = [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ]
        _, _, yaw = euler_from_quaternion(ori)
        pts = np.array([[c[0], c[1], -self.lidar_z_offset] for c in cones])
        M4 = self._transform_to_matrix(transform)
        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        transformed = (M4 @ pts_h.T).T
        return [(transformed[i, 0], transformed[i, 1], cones[i][2]) for i in range(len(cones))], yaw
    
    # --- Visualization helpers (must live in SceneGenerator) ---
    def _load_pointcloud_o3d(self, pcd: np.ndarray) -> o3d.geometry.PointCloud:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pcd[:, :3])
        zs = pcd[:, 2]
        nz = (zs - zs.min()) / (zs.ptp() + 1e-8)
        pc.colors = o3d.utility.Vector3dVector(plt.get_cmap('cividis')(nz)[:, :3])
        return pc

    def _create_bbox_from_label(self, center: np.ndarray, size: Tuple[float,float,float], color: Tuple[float,float,float]):
        hs = np.array(size) / 2.0
        aabb = o3d.geometry.AxisAlignedBoundingBox(center - hs, center + hs)
        box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
        box.colors = o3d.utility.Vector3dVector([color] * len(box.lines))
        return box

    def _visualize_scene(self, pcd: np.ndarray, label_path: str):
        """
        Open an Open3D window showing the raw pointcloud plus the KITTI boxes
        read from label_path.
        """
        try:
            pc = self._load_pointcloud_o3d(pcd)
            geoms = [pc]
            cmap = {
            'yellow_cone': (1,1,0),
            'blue_cone':   (0,0,1),
            'orange_cone': (1,0.5,0),
            'large_orange_cone':    (0.5,0.0,0.5),
            }

            with open(label_path, 'r') as lf:
                for ln in lf:
                    parts = ln.split()
                    if len(parts) < 15: 
                        continue
                    lbl = parts[0]
                    h, w, l = map(float, parts[8:11])
                    x, y, z = map(float, parts[11:14])
                    col = cmap.get(lbl, (0.5,0.5,0.5))
                    geoms.append(self._create_bbox_from_label(np.array([x,y,z]), (l,w,h), col))
                    sph = o3d.geometry.TriangleMesh.create_sphere(0.05)
                    sph.translate((x,y,z))
                    sph.paint_uniform_color(col)
                    geoms.append(sph)

            vis = o3d.visualization.Visualizer()
            vis.create_window("Preview", width=900, height=600)
            for g in geoms:
                vis.add_geometry(g)
            opt = vis.get_render_option()
            opt.point_size = 2.0
            vis.run()
            vis.destroy_window()
        except Exception as e:
            print("[Open3D Error]", e)


    def _read_data(
        self,
        global_cones: List[Tuple[float, float, int]]
    ) -> Generator[
        Tuple[np.ndarray, List[Tuple[float, float, int, Tuple[float,float,float]]], Dict[str, Any], Optional[np.ndarray]],
        None,
        None
    ]:
        bag = rosbag.Bag(self.selected_file, 'r')
        vx = vy = yawrate = 0.0
        lap_count = None

        for i, (topic, msg, _) in enumerate(bag.read_messages()):
            if topic in ["/tf", "/tf_static"]:
                self._update_tf(msg)
                continue
            if topic == LAP_COUNT_TOPIC:
                lap_count = msg.data
                continue
            if topic == CAR_STATE_TOPIC:
                if isinstance(msg, Odometry):
                    vx      = msg.twist.twist.linear.x
                    vy      = msg.twist.twist.linear.y
                    yawrate = msg.twist.twist.angular.z
                continue
            if topic != POINTCLOUD_TOPIC:
                continue

            pc = pc2.read_points(
                msg,
                field_names=('x','y','z','intensity','timestamp'),
                skip_nans=True
            )
            arr = np.array(list(pc), dtype=np.float64)
            if self.ego_motion_compensate and arr.shape[1] >= 5:
                arr = self._compensate_point_cloud(arr, msg.header.stamp)

            tfc = self._lookup_transform_chain(CONE_FRAME, msg.header.frame_id, msg.header.stamp)
            if not tfc:
                print(f"[ERROR] Could not find transform from {msg.header.frame_id} to {CONE_FRAME} at {msg.header.stamp.to_sec():.3f}")
                continue

            cones_all, yaw = self._get_local_cones(global_cones, tfc.transform, arr)

            if global_cones:
                print(f"[DEBUG] Example original cone (global): {global_cones[0]}")
            if cones_all:
                print(f"[DEBUG] Transformed cone (local): {cones_all[0]}")

            visible = []
            for x, y, c in cones_all:
                if x <= 0:
                    continue
                r = np.hypot(x, y)
                if not (self.min_range <= r <= self.max_range):
                    continue
                angle = np.degrees(np.arctan2(y, x))
                if not (self.min_angle <= angle <= self.max_angle):
                    continue
                dims = DIMENSIONS.get(c)
                if dims is None:
                    continue
                dims = tuple(d * (1 + self.expansion_coeff) for d in dims)
                visible.append((x, y, c, dims))

            odom = {
                'timestamp': msg.header.stamp.to_sec(),
                'x': tfc.transform.translation.x,
                'y': tfc.transform.translation.y,
                'z': tfc.transform.translation.z,
                'yaw': yaw,
                'vx': vx,
                'vy': vy,
                'yawrate': yawrate,
                'lap': lap_count
            }

            img = self._get_closest_image(msg.header.stamp)

            yield arr, visible, odom, img
        bag.close()


    def gen_data(self, global_cones: List[Tuple[float, float, int]]):
        if not global_cones:
            messagebox.showinfo("Info", "No cones to export.")
            return
        if not messagebox.askyesno("Export", "Proceed?"):
            return

        folder = filedialog.askdirectory(title="Choose output folder")
        if not folder:
            return
        for d in ['points','labels','images']:
            os.makedirs(os.path.join(folder, d), exist_ok=True)

        meta = {
            'tool': 'generate_scene_with_yolo.py',
            'version': '0.11-yolo-integration',
            'generated_on': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': os.path.basename(self.selected_file),
            'user': gt.getuser(),
        }

        fid = 0
        data = []

        for idx, (pcd, cones, odom, image) in enumerate(self._read_data(global_cones)):
            if idx < 3 or (idx > 3 and (idx - 3) % self.label_every != 0):
                continue

            fn = f"{fid:07d}"
            dets: List[Tuple[float, ...]] = []

            # Save pointcloud
            pcd.astype(np.float32).tofile(f"{folder}/points/{fn}.bin")

            # Save raw image
            if image is not None:
                cv2.imwrite(f"{folder}/images/{fn}.png", image)
            else:
                print(f"[DEBUG] No image found for frame {fn}, skipping image save and YOLO")

            # Run YOLO
            scale_x, scale_y = 1.0, 1.0
            if image is not None and self.yolo_weights:
                print(f"[DEBUG] Running YOLO on frame {fn}")
                img_h, img_w = image.shape[:2]
                scale_x = img_w / self.yolo_img_size
                scale_y = img_h / self.yolo_img_size
                dets = self.yolo_model.inference(image).tolist()
                print(f"[DEBUG] YOLO returned {len(dets)} detections")

            # Save KITTI labels
            label_path = f"{folder}/labels/{fn}.txt"
            with open(label_path, 'w') as lf:
                for x, y, c, dims in cones:
                    h, w, l = dims[2], dims[0], dims[1]
                    z = dims[2]/2 - self.lidar_z_offset
                    print(f"[DEBUG] Writing KITTI label with cone pos: ({x:.2f}, {y:.2f}, {z:.2f})")
                    lf.write(
                        f"{LABEL_NAME[c]} 0.00 0 0.00 0.00 0.00 0.00 0.00 "
                        f"{h:.3f} {w:.3f} {l:.3f} "
                        f"{x:.3f} {y:.3f} {z:.3f} 0.00\n"
                    )
                for *box, conf, cls_id in dets:
                    x1, y1, x2, y2 = map(float, box)
                    x1 *= scale_x
                    y1 *= scale_y
                    x2 *= scale_x
                    y2 *= scale_y
                    cls_id = int(cls_id)
                    cls_name = (self.class_labels[cls_id]
                                if cls_id < len(self.class_labels)
                                else f"cls{cls_id}")
                    lf.write(
                        f"{cls_name} 0.00 0 {x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f} "
                        "0.00 0.00 0.00 0.00 0.00 0.00 0.00\n"
                    )

            if self.visualize:
                self._visualize_scene(pcd, label_path)

            if self.visualize and image is not None and self.yolo_weights:
                vis = image.copy()
                for *box, conf, cls in dets:
                    x1, y1, x2, y2 = map(float, box)
                    x1 *= scale_x
                    y1 *= scale_y
                    x2 *= scale_x
                    y2 *= scale_y
                    cls_id = int(cls)
                    class_name = (self.class_labels[cls_id]
                                if cls_id < len(self.class_labels)
                                else f"cls{cls_id}")
                    color = VIS_COLORS.get(class_name, (1, 1, 1))
                    bgr = tuple(int(255 * c) for c in color[::-1])
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), bgr, 2)
                    cv2.imshow("YOLO Preview", vis)
                    cv2.waitKey(0)
                    cv2.destroyWindow("YOLO Preview")

            entry = {
                'id': fid,
                'odom': odom,
                'pointcloud': {'file': f'points/{fn}.bin'},
                'labels': {'file': f'labels/{fn}.txt'},
            }
            if image is not None:
                entry['image'] = {'file': f'images/{fn}.png'}
            with open(f"{folder}/metadata.json", 'a') as mf:
                mf.write(json.dumps(entry) + ",\n")

            fid += 1

        with open(f"{folder}/metadata.json", 'r+') as mf:
            lines = mf.readlines()
        with open(f"{folder}/metadata.json", 'w') as mf:
            mf.write('{\n  "info": ' + json.dumps(meta, indent=2) + ',\n')
            mf.write('  "data": [\n' + ''.join(lines).rstrip(',\n') + '\n  ]\n}')

        messagebox.showinfo("Done", f"Saved {fid} frames to {folder}\n\n"
                                    "Next: run 'label_scenes.py' to refine per-frame boxes.")


class MapEditor:
    def __init__(
        self,
        label_every: int = 400,
        min_range: float = 2.0,
        max_range: float = 20.0,
        expansion_coeff: float = 0.0,
        centroid_correction: bool = False,
        cluster_radius: float = 0.5,
        min_cluster_points: int = 5,
        min_angle: float = -70.0,
        max_angle: float = 70.0,
        yolo_weights: str = "",
        yolo_classes: str = "",
        yolo_img_size: int = 640,
        yolo_conf_thresh: float = 0.35,
        yolo_iou_thresh: float = 0.45,
        yolo_device: str = "cpu",
        yolo_half: bool = False,
        max_img_lag: float = 0.1,
    ):
        # exactly your original MapEditor.__init__, no removals
        self.cones: List[Tuple[float, float, int]] = []
        self.selected_cones: Set[Tuple[float, float, int]] = set()
        self.selected_file: str = ""
        self.label_every = label_every
        self.min_range = min_range
        self.max_range = max_range
        self.expansion_coeff = expansion_coeff
        self.centroid_correction = centroid_correction
        self.cluster_radius = cluster_radius
        self.min_cluster_points = min_cluster_points
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.yolo_weights = yolo_weights
        self.yolo_classes = yolo_classes
        self.yolo_img_size = yolo_img_size
        self.yolo_conf_thresh = yolo_conf_thresh
        self.yolo_iou_thresh = yolo_iou_thresh
        self.yolo_device = yolo_device
        self.yolo_half = yolo_half
        self.max_img_lag = max_img_lag

        self.root = tk.Tk()
        self.root.title("Map Editor")

        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.visualize = tk.BooleanVar(value=False)

        self.setup_ui()

class MapEditor:
    def __init__(
        self,
        label_every: int = 400,
        min_range: float = 2.0,
        max_range: float = 20.0,
        expansion_coeff: float = 0.0,
        centroid_correction: bool = False,
        cluster_radius: float = 0.5,
        min_cluster_points: int = 5,
        min_angle: float = -70.0,
        max_angle: float = 70.0,
        # YOLO args
        yolo_weights: str = "",
        yolo_classes: str = "",
        yolo_img_size: int = 640,
        yolo_conf_thresh: float = 0.35,
        yolo_iou_thresh: float = 0.45,
        yolo_device: str = "cpu",
        yolo_half: bool = False,
        max_img_lag: float = 0.1,
    ):
        self.cones: List[Tuple[float, float, int]] = []
        self.selected_cones: Set[Tuple[float, float, int]] = set()
        self.selected_file: str = ""
        self.label_every = label_every
        self.min_range = min_range
        self.max_range = max_range
        self.expansion_coeff = expansion_coeff
        self.centroid_correction = centroid_correction
        self.cluster_radius = cluster_radius
        self.min_cluster_points = min_cluster_points
        self.min_angle = min_angle
        self.max_angle = max_angle
        # YOLO args
        self.yolo_weights = yolo_weights
        self.yolo_classes = yolo_classes
        self.yolo_img_size = yolo_img_size
        self.yolo_conf_thresh = yolo_conf_thresh
        self.yolo_iou_thresh = yolo_iou_thresh
        self.yolo_device = yolo_device
        self.yolo_half = yolo_half
        self.max_img_lag = max_img_lag
        # RGBA mapping index → (r,g,b,a)
        # RGBA mapping index → (r,g,b,a)
        self.color_map = {
            UNKNOWN_C: (0.5, 0.5, 0.5, 0.8),  # gray for any unknown
            BLUE_C:    (0.0, 0.0, 1.0, 0.8),
            YELLOW_C:  (1.0, 1.0, 0.0, 0.8),
            ORANGE_C:  (1.0, 0.5, 0.0, 0.8),
            BIG_C:     (0.5, 0.0, 0.5, 0.8),
        }

        self.root = tk.Tk()
        self.root.title("Map Editor")

        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.visualize = tk.BooleanVar(value=False)

        self.setup_ui()

    def setup_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side="top", fill="x", padx=10, pady=5)
        inner_top = tk.Frame(top_frame)
        inner_top.pack(anchor="center")
        tk.Button(inner_top, text="Open File", command=self.open_file).pack(side="left")
        self.selected_file_label = tk.Label(inner_top, text=self.selected_file, anchor="w")
        self.selected_file_label.pack(side="left", padx=10)

        self.canvas.get_tk_widget().pack(pady=10)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(pady=10)
        for text, cmd in [
            ("Remove Selected", self.remove_selected_cones),
            ("Export Dataset",  self.gen_data),
            ("Set Yellow",      lambda: self.set_color(YELLOW_C)),
            ("Set Blue",        lambda: self.set_color(BLUE_C)),
            ("Set Big",         lambda: self.set_color(BIG_C)),
            ("Set Orange",      lambda: self.set_color(ORANGE_C)),
            ("Close",           self.quit_all)
        ]:
            tk.Button(bottom_frame, text=text, command=cmd).pack(side="left", padx=5)
        tk.Checkbutton(bottom_frame, text="Enable Visualization", variable=self.visualize).pack(side="right", padx=10)

    def quit_all(self):
        self.root.destroy()
        os._exit(0)

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[("ROS Bags", "*.bag")])
        if not path:
            return
        self.selected_file = path
        self.selected_file_label.config(text=os.path.basename(path))
        bag = rosbag.Bag(path, 'r')
        md = {}
        for _, msg, _ in bag.read_messages(topics=[SLAM_MAP_TOPIC]):
            for m in msg.markers:
                if m.action == 0:
                    md[m.id] = m
                elif m.action == 2 and m.id in md:
                    del md[m.id]
        bag.close()

        self.cones.clear()
        for m in md.values():
            x, y = m.pose.position.x, m.pose.position.y
            r, g, b = m.color.r, m.color.g, m.color.b
            if (r, g, b) == (0, 0, 1):
                c = BLUE_C
            elif (r, g, b) == (1, 1, 0):
                c = YELLOW_C
            elif (r, g, b) == (0.5, 0.5, 0.5):
                c = ORANGE_C
            else:
                c = UNKNOWN_C
            self.cones.append((x, y, c))
        print(f"[Info] Loaded {len(self.cones)} landmarks")
        self.plot_cones()

    def plot_cones(self):
        self.ax.clear()
        if self.cones:
            xs, ys, cs = zip(*self.cones)
            cols = [ self.color_map.get(c, (0.5,0.5,0.5,1.0)) for c in cs ]
            self.ax.scatter(xs, ys, c=cols, s=30)
        if self.selected_cones:
            sx, sy, _ = zip(*self.selected_cones)
            self.ax.scatter(sx, sy, color="red", s=50, marker="x")
        self.ax.set_xlabel("X"); self.ax.set_ylabel("Y")
        self.ax.set_title("Cone Map")
        self.canvas.draw()

    def on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = event.xdata, event.ydata
        best, md = None, float("inf")
        for pt in self.cones:
            d = np.hypot(x - pt[0], y - pt[1])
            if d < md:
                best, md = pt, d
        if best and md < 1.5:
            if best in self.selected_cones:
                self.selected_cones.remove(best)
            else:
                self.selected_cones.add(best)
            self.plot_cones()

    def set_color(self, c):
        if not self.selected_cones:
            messagebox.showinfo("Info", "Select cones first.")
            return
        updated = []
        for pt in self.selected_cones:
            self.cones.remove(pt)
            updated.append((pt[0], pt[1], c))
        self.cones.extend(updated)
        self.selected_cones.clear()
        self.plot_cones()

    def remove_selected_cones(self):
        if not self.selected_cones:
            messagebox.showinfo("Info", "No cones selected.")
            return
        self.cones = [c for c in self.cones if c not in self.selected_cones]
        self.selected_cones.clear()
        self.plot_cones()

    def gen_data(self):
        sg = SceneGenerator(
            self.selected_file,
            self.label_every,
            ego_motion_compensate=True,
            lidar_z_offset=1.15,
            visualize=self.visualize.get(),
            lidar_freq=20.0,
            min_range=self.min_range,
            max_range=self.max_range,
            expansion_coeff=self.expansion_coeff,
            centroid_correction=self.centroid_correction,
            cluster_radius=self.cluster_radius,
            min_cluster_points=self.min_cluster_points,
            min_angle=self.min_angle,
            max_angle=self.max_angle,
            # YOLO args
            yolo_weights=self.yolo_weights,
            yolo_classes=self.yolo_classes,
            yolo_img_size=self.yolo_img_size,
            yolo_conf_thresh=self.yolo_conf_thresh,
            yolo_iou_thresh=self.yolo_iou_thresh,
            yolo_device=self.yolo_device,
            yolo_half=self.yolo_half,
            max_img_lag=self.max_img_lag,
        )
        sg.gen_data(self.cones)
        messagebox.showinfo(
            "Done",
            f"Export completed. You can now run 'label_scenes.py' to refine per-frame boxes."
        )


    def run(self):
        self.root.mainloop()

def main():
    rospy.init_node("generate_scene_gui", anonymous=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-every",        type=int,   default=20)
    parser.add_argument("--min-range",          type=float, default=2.0)
    parser.add_argument("--max-range",          type=float, default=40.0)
    parser.add_argument("--expansion-coeff",    type=float, default=0.1)
    parser.add_argument("--centroid-correction",action="store_true")
    parser.add_argument("--cluster-radius",     type=float, default=0.5,
                        help="radius around expected cone for cluster centroid")
    parser.add_argument("--min-cluster-points", type=int,   default=5,
                        help="minimum points to accept a cluster")
    parser.add_argument("--min-angle",          type=float, default=-75.0,
                        help="min azimuth angle (deg)")
    parser.add_argument("--max-angle",          type=float, default=75.0,
                        help="max azimuth angle (deg)")
    # YOLO CLI args
    parser.add_argument("--yolo-weights",       type=str,   default="weights/yolov11m.pt")
    parser.add_argument("--yolo-classes",       type=str,   default="class_labels/fscoco.txt")
    parser.add_argument("--yolo-img-size",      type=int,   default=640)
    parser.add_argument("--yolo-conf-thresh",   type=float, default=0.30)
    parser.add_argument("--yolo-iou-thresh",    type=float, default=0.5)
    parser.add_argument("--yolo-device",        type=str,   default="cpu")
    parser.add_argument("--yolo-half",          action="store_true")
    parser.add_argument("--max-img-lag",        type=float, default=0.05,
                        help="max allowable lag (s) to sync image")
    args = parser.parse_args()

    editor = MapEditor(
        label_every=args.label_every,
        min_range=args.min_range,
        max_range=args.max_range,
        expansion_coeff=args.expansion_coeff,
        centroid_correction=args.centroid_correction,
        cluster_radius=args.cluster_radius,
        min_cluster_points=args.min_cluster_points,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
        yolo_weights=args.yolo_weights,
        yolo_classes=args.yolo_classes,
        yolo_img_size=args.yolo_img_size,
        yolo_conf_thresh=args.yolo_conf_thresh,
        yolo_iou_thresh=args.yolo_iou_thresh,
        yolo_device=args.yolo_device,
        yolo_half=args.yolo_half,
        max_img_lag=args.max_img_lag,
    )
    editor.run()

if __name__ == "__main__":
    main()
