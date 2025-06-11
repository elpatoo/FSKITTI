#!/usr/bin/env python3
import os, glob
import numpy as np
import tkinter as tk
import cv2
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ORIG_W, ORIG_H = 2048, 1536  # original image resolution for scaling

def parse_full_label(path):
    entries = []
    with open(path, 'r') as f:
        for ln in f:
            p = ln.split()
            if len(p) < 15: continue
            typ = p[0]
            left, top, right, bottom = map(float, p[4:8])
            h, w, l = map(float, p[8:11])
            x, y, z = map(float, p[11:14])
            entries.append({
                'type': typ,
                'left': left, 'top': top,
                'right': right, 'bottom': bottom,
                'h': h, 'w': w, 'l': l,
                'x_cam': x, 'y_cam': y, 'z_cam': z  # camera frame
            })
    return entries

def parse_calib_file(calib_path):
    """Parse KITTI calib file into a dict of matrix entries."""
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if ':' not in line: continue
            key, val = line.strip().split(':', 1)
            data = np.fromstring(val, sep=' ')
            if key.startswith('Tr_'):
                calib[key] = data.reshape(3, 4)
            elif key == 'R0_rect':
                calib[key] = data.reshape(3, 3)
    return calib

def transform_cam_to_lidar(xyz_cam, Tr_velo_to_cam):
    """Transform 3D points from camera to LiDAR frame."""
    T = np.eye(4)
    T[:3, :4] = Tr_velo_to_cam
    T_inv = np.linalg.inv(T)
    xyz_cam_hom = np.hstack([xyz_cam, np.ones((xyz_cam.shape[0], 1))])
    xyz_lidar_hom = (T_inv @ xyz_cam_hom.T).T
    return xyz_lidar_hom[:, :3]

class MatchedVisualizer:
    def __init__(self, folder):
        self.folder = folder
        label_dir = os.path.join(folder, 'label_2')
        self.frame_ids = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(label_dir, '*.txt'))
        )
        if not self.frame_ids:
            messagebox.showerror("Error", f"No labels found in {label_dir}")
            raise SystemExit
        self.idx = 0

        self._zpcd_xlim = (-20, 20)
        self._zpcd_ylim = (-20, 20)

        self._load_frame()

        self.root = tk.Tk()
        self.root.title("Matched Cones Viewer")

        self.fig = Figure(figsize=(12,6))
        self.ax_pcd = self.fig.add_subplot(121)
        self.ax_img = self.fig.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

        nav = tk.Frame(self.root)
        nav.pack(side=tk.BOTTOM, fill=tk.X, pady=4)
        tk.Frame(nav).pack(side=tk.LEFT, expand=True)
        for txt, cmd in [("Prev", self._on_prev),
                         ("Next", self._on_next),
                         ("Quit", self.root.quit)]:
            tk.Button(nav, text=txt, command=cmd).pack(side=tk.LEFT, padx=8)
        tk.Frame(nav).pack(side=tk.LEFT, expand=True)

        self._draw()
        tk.mainloop()

    def _load_frame(self):
        fid = self.frame_ids[self.idx]

        pts = np.fromfile(
            os.path.join(self.folder, 'velodyne', f"{fid}.bin"),
            dtype=np.float32
        ).reshape(-1,5)
        self.pcd = pts[:,:3]

        img = cv2.imread(os.path.join(self.folder, 'image_2', f"{fid}.png"))
        if img is None:
            messagebox.showerror("Error", f"Image not found: {fid}.png")
            raise SystemExit
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        entries = parse_full_label(os.path.join(self.folder, 'label_2', f"{fid}.txt"))
        entries = [e for e in entries if (e['right'] - e['left']) > 1e-3 and e['h'] > 1e-3]

        if entries:
            calib_file = os.path.join(self.folder, 'calib', f"{fid}.txt")
            calib = parse_calib_file(calib_file)
            Tr_velo_to_cam = calib['Tr_velo_to_cam']

            xyz_cam = np.array([[e['x_cam'], e['y_cam'], e['z_cam']] for e in entries])
            xyz_lidar = transform_cam_to_lidar(xyz_cam, Tr_velo_to_cam)

            for i, e in enumerate(entries):
                e['x'], e['y'], e['z'] = xyz_lidar[i]
        else:
            # No valid matched objects
            for e in entries:
                e['x'], e['y'], e['z'] = 0.0, 0.0, 0.0  # optional default

        self.matched = entries

    def _draw(self):
        self.ax_pcd.clear()
        rx, ry = -self.pcd[:,1], self.pcd[:,0]
        self.ax_pcd.scatter(rx, ry, s=1)

        fov, R = np.radians(32.5), 40
        for a in (-fov, fov):
            ex, ey = R*np.cos(a), R*np.sin(a)
            self.ax_pcd.plot([-ey,0], [ex,0], color='red', alpha=0.3, linewidth=2)

        cmap = plt.cm.get_cmap('tab20', max(len(self.matched),1))
        for i, e in enumerate(self.matched):
            col = cmap(i)[:3]
            x, y, w, l = e['x'], e['y'], e['w'], e['l']
            rect = plt.Rectangle(
                (-y - w/2, x - l/2), w, l,
                fill=False, edgecolor=col, linewidth=8
            )
            self.ax_pcd.add_patch(rect)

        self.ax_pcd.set_xlim(*self._zpcd_xlim)
        self.ax_pcd.set_ylim(*self._zpcd_ylim)
        self.ax_pcd.set_xlabel('X')
        self.ax_pcd.set_ylabel('Y')
        self.ax_pcd.axis('equal')

        self.ax_img.clear()
        overlay = self.img.copy()
        h, w = overlay.shape[:2]
        sx, sy = w/ORIG_W, h/ORIG_H

        for i, e in enumerate(self.matched):
            col = tuple(int(c*255) for c in cmap(i)[:3])
            x1, y1 = int(e['left']*sx), int(e['top']*sy)
            x2, y2 = int(e['right']*sx), int(e['bottom']*sy)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 8)

        self.ax_img.imshow(overlay)
        self.ax_img.axis('off')
        self.canvas.draw()

    def _on_scroll(self, event):
        if event.inaxes == self.ax_pcd and event.xdata is not None:
            factor = 1.2 if getattr(event, 'step', 1) > 0 else 1/1.2
            x0, x1 = self._zpcd_xlim
            y0, y1 = self._zpcd_ylim
            cx, cy = event.xdata, event.ydata
            self._zpcd_xlim = (cx - (cx - x0) * factor, cx + (x1 - cx) * factor)
            self._zpcd_ylim = (cy - (cy - y0) * factor, cy + (y1 - cy) * factor)
            self._draw()

    def _on_next(self):
        self.idx = (self.idx + 1) % len(self.frame_ids)
        self._load_frame()
        self._draw()

    def _on_prev(self):
        self.idx = (self.idx - 1) % len(self.frame_ids)
        self._load_frame()
        self._draw()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="KITTI folder with image_2/, label_2/, velodyne/, calib/")
    args = p.parse_args()
    MatchedVisualizer(args.folder)
