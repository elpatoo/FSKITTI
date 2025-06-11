#!/usr/bin/env python3
import os
import glob
import numpy as np
import tkinter as tk
import cv2
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import messagebox
from tkinter import simpledialog

ORIG_W, ORIG_H = 2048, 1536

def parse_label_file(path):
    slam_entries, yolo_entries = [], []
    with open(path, 'r') as f:
        for idx, ln in enumerate(f):
            parts = ln.strip().split()
            if len(parts) < 14:
                continue
            typ = parts[0]
            if len(parts) >= 15:
                left, top, right, bottom = map(float, parts[4:8])
                h, w, l = map(float, parts[8:11])
                x, y, z = map(float, parts[11:14])
            else:
                left, top, right, bottom = map(float, parts[3:7])
                h, w, l = map(float, parts[7:10])
                x, y, z = map(float, parts[10:13])
            slam_entries.append({
                'id': idx, 'type': typ,
                'h': h, 'w': w, 'l': l,
                'x': x, 'y': y, 'z': z
            })
            if (right - left) > 1 and (bottom - top) > 1:
                yolo_entries.append({
                    'id': idx, 'type': typ,
                    'left': left, 'top': top,
                    'right': right, 'bottom': bottom
                })
    return slam_entries, yolo_entries

def write_label_file(path, slam_entries, yolo_entries, matched_entries):
    lines = []

    def is_empty(entry):
        # true if all 2D + 3D fields are zero or missing
        return all(entry.get(k, 0.0) == 0.0
                   for k in ('left','top','right','bottom','h','w','l','x','y','z'))

    # 1) SLAM-only: real 3D, dummy 2D
    for e in slam_entries:
        if is_empty(e):
            continue
        lines.append(
            f"{e['type']} 0.00 0 0.00 "               # class, trunc, occ, alpha
            f"0.00 0.00 0.00 0.00 "                   # dummy 2D
            f"{e['h']:.3f} {e['w']:.3f} {e['l']:.3f} "  # 3D dims
            f"{e['x']:.3f} {e['y']:.3f} {e['z']:.3f} "  # 3D loc
            "0.00"                                     # ry
        )

    # 2) Matched: real 2D + real 3D, with resolved class
    for e in matched_entries:
        lines.append(
            f"{e['type']} 0.00 0 0.00 "
            f"{e['left']:.2f} {e['top']:.2f} {e['right']:.2f} {e['bottom']:.2f} "  # real 2D
            f"{e['h']:.3f} {e['w']:.3f} {e['l']:.3f} "                            # real 3D dims
            f"{e['x']:.3f} {e['y']:.3f} {e['z']:.3f} "                            # real 3D loc
            "0.00"
        )

    # 3) YOLO-only: real 2D, dummy 3D
    for e in yolo_entries:
        if is_empty(e):
            continue
        lines.append(
            f"{e['type']} 0.00 0 0.00 "
            f"{e['left']:.2f} {e['top']:.2f} {e['right']:.2f} {e['bottom']:.2f} "  # real 2D
            "0.00 0.00 0.00 "                         # dummy 3D dims
            "0.00 0.00 0.00 "                         # dummy 3D loc
            "0.00"
        )

    with open(path, 'w') as f:
        f.write('\n'.join(lines))



class FrameLabeler:
    def __init__(self, folder):
        self.folder = folder
        self.frame_ids = sorted(os.path.splitext(os.path.basename(p))[0]
                                for p in glob.glob(os.path.join(folder, 'labels','*.txt')))
        self.idx = 0
        self.selected_3d = None
        self.selected_2d = None
        self.dragging_3d = False
        self.zoom_pcd = False
        self.zoom_img = False
        # pan state for point-cloud
        self.panning_pcd = False
        self.pan_start   = (0, 0)
        self.pan_xlim0   = (0, 0)
        self.pan_ylim0   = (0, 0)
        self.matched_entries = []
        self.match_history = []

        self._load_frame()

        self.root = tk.Tk()
        self.root.title("Label Editor")
        self.fig = Figure(figsize=(12,6))
        self.ax_pcd = self.fig.add_subplot(121)
        self.ax_img = self.fig.add_subplot(122)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.get_tk_widget().config(cursor='arrow')
        NavigationToolbar2Tk(self.canvas, self.root).update()

        tk.Button(self.root, text="Save + Next", command=self._on_next).pack(fill=tk.X)
        tk.Button(self.root, text="Match Cones", command=self._on_match).pack(fill=tk.X)
        tk.Button(self.root, text="Undo Last Match",   command=self._on_undo).pack(fill=tk.X)
        tk.Button(self.root, text="Reset Frame",        command=self._on_reset).pack(fill=tk.X)

        self.canvas.mpl_connect("button_press_event",   self._on_click)
        self.canvas.mpl_connect("motion_notify_event",  self._on_drag)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event",         self._on_scroll)

        self._draw()
        tk.mainloop()

    def _load_frame(self):
        fid = self.frame_ids[self.idx]
        pts = np.fromfile(os.path.join(self.folder,'points',f"{fid}.bin"), dtype=np.float32).reshape(-1,5)
        self.pcd = pts[:,:3]
        img = cv2.imread(os.path.join(self.folder,'images',f"{fid}.png"))
        if img is None:
            raise FileNotFoundError(f"Image not found: {fid}.png")
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = os.path.join(self.folder,'labels',f"{fid}.txt")
        self.slam_entries, self.yolo_entries = parse_label_file(lab)

    def _get_color(self, typ):
        return {
            'yellow_cone': 'yellow',
            'blue_cone': 'blue',
            'orange_cone': 'orange',
            'big_cone': 'purple'
        }.get(typ, 'gray')

    def _get_bgr_color(self, typ):
        return {
            'yellow_cone': (255, 255, 0),
            'blue_cone':   (0, 0, 255),
            'orange_cone': (255, 165, 0),
            'big_cone':    (255, 0, 255)
        }.get(typ, (192, 192, 192))

    def _draw(self):
        self.ax_pcd.clear()
        rotated_x = -self.pcd[:,1]
        rotated_y = self.pcd[:,0]
        self.ax_pcd.scatter(rotated_x, rotated_y, s=1)

        fov_deg = 32.5
        fov_rad = np.radians(fov_deg)
        radius = 40.0

        for angle in [-fov_rad, fov_rad]:
            end_x = radius * np.cos(angle)
            end_y = radius * np.sin(angle)
            rx = -end_y
            ry = end_x
            self.ax_pcd.plot([0, rx], [0, ry], color='red', linewidth=2, alpha=0.3)

        for i, e in enumerate(self.slam_entries):
            if i == self.selected_3d:
                continue
            x, y, l, w = e['x'], e['y'], e['l'], e['w']
            rx, ry = -y, x
            rl, rw = w, l
            col = self._get_color(e['type'])
            self.ax_pcd.add_patch(
                plt.Rectangle((rx - rl/2, ry - rw/2), rl, rw,
                              fill=False, edgecolor=col, linewidth=2)
            )

        # Draw selected 3D in red
        if self.selected_3d is not None:
            e = self.slam_entries[self.selected_3d]
            x, y, l, w = e['x'], e['y'], e['l'], e['w']
            rx, ry = -y, x
            rl, rw = w, l
            self.ax_pcd.add_patch(
                plt.Rectangle((rx - rl/2, ry - rw/2), rl, rw,
                              fill=False, edgecolor='red', linewidth=3)
            )

        self.ax_pcd.set_xlabel('X'); self.ax_pcd.set_ylabel('Y'); self.ax_pcd.axis('equal')
        if self.zoom_pcd:
            self.ax_pcd.set_xlim(self._zpcd_xlim)
            self.ax_pcd.set_ylim(self._zpcd_ylim)
        else:
            self.ax_pcd.set_xlim(-30, 30)
            self.ax_pcd.set_ylim(25, 25)

        self.ax_img.clear()
        h, w = self.img.shape[:2]
        sx, sy = w / ORIG_W, h / ORIG_H
        overlay = self.img.copy()
        selected = self.selected_2d

        for i, e in enumerate(self.yolo_entries):
            if i == selected:
                continue
            x1 = int(e['left'] * sx)
            y1 = int(e['top'] * sy)
            x2 = int(e['right'] * sx)
            y2 = int(e['bottom'] * sy)
            col = self._get_bgr_color(e['type'])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 4)

        if selected is not None:
            e = self.yolo_entries[selected]
            x1 = int(e['left'] * sx)
            y1 = int(e['top'] * sy)
            x2 = int(e['right'] * sx)
            y2 = int(e['bottom'] * sy)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 4)

        self.ax_img.imshow(overlay)
        self.ax_img.axis('off')
        if self.zoom_img:
            self.ax_img.set_xlim(self._zimg_xlim)
            self.ax_img.set_ylim(self._zimg_ylim)

        self.canvas.draw()

    def _on_click(self, event):

        # — start pan on right-click in PCD view —
        if event.button == 3 and event.inaxes == self.ax_pcd \
           and event.xdata is not None and event.ydata is not None:
            self.panning_pcd = True
            self.pan_start   = (event.xdata, event.ydata)
            self.pan_xlim0   = self.ax_pcd.get_xlim()
            self.pan_ylim0   = self.ax_pcd.get_ylim()
            return

        if event.inaxes == self.ax_pcd and event.xdata is not None and event.ydata is not None:
            click_x = event.ydata
            click_y = -event.xdata
            d = [np.hypot(click_x - e['x'], click_y - e['y']) for e in self.slam_entries]
            i = int(np.argmin(d))
            if d[i] < 1.0:
                self.selected_3d = i
                self.dragging_3d = True

        elif event.inaxes == self.ax_img and event.xdata is not None and event.ydata is not None:
            x, y = event.xdata, event.ydata
            h, w = self.img.shape[:2]
            sx, sy = w / ORIG_W, h / ORIG_H
            for i, e in enumerate(self.yolo_entries):
                x1 = e['left'] * sx
                y1 = e['top'] * sy
                x2 = e['right'] * sx
                y2 = e['bottom'] * sy
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_2d = i
                    break
            else:
                self.selected_2d = None

            print(f"Clicked image at: ({x:.1f}, {y:.1f})")
            if self.selected_2d is not None:
                print(f"Selected 2D box index: {self.selected_2d}, type: {self.yolo_entries[self.selected_2d]['type']}")
            else:
                print("No 2D box selected.")

        self._draw()

    def _on_drag(self, event):
        # — if panning, update limits and redraw —
        if self.panning_pcd and event.inaxes == self.ax_pcd \
           and event.xdata is not None and event.ydata is not None:
            dx = self.pan_start[0] - event.xdata
            dy = self.pan_start[1] - event.ydata
            x0, x1 = self.pan_xlim0
            y0, y1 = self.pan_ylim0
            self.ax_pcd.set_xlim(x0 + dx, x1 + dx)
            self.ax_pcd.set_ylim(y0 + dy, y1 + dy)
            self.canvas.draw()
            return

        # — otherwise handle 3D-box dragging as before —
        if self.dragging_3d and self.selected_3d is not None and event.inaxes == self.ax_pcd:
            # update the SLAM entry in memory
            self.slam_entries[self.selected_3d]['x'] = event.ydata
            self.slam_entries[self.selected_3d]['y'] = -event.xdata
            self._draw()

        
    def _on_next(self):
        fid = self.frame_ids[self.idx]
        label_path = os.path.join(self.folder, 'labels', f"{fid}.txt")

        # Save current frame: only SLAM-only, matched, and YOLO-only entries as above
        write_label_file(
            label_path,
            slam_entries    = self.slam_entries,
            yolo_entries    = self.yolo_entries,
            matched_entries = self.matched_entries
        )

        # Clear merges & history so they don't carry over
        self.matched_entries.clear()
        self.match_history.clear()

        # Advance to next frame
        self.idx += 1
        if self.idx >= len(self.frame_ids):
            messagebox.showinfo("Done", "Reached end of frames.")
            self.root.quit()
            return

        # Load and redraw
        self._load_frame()
        self.selected_3d = None
        self.selected_2d = None
        self._draw()


    def _on_release(self, event):
        # stop both 3D‐drag and pan; preserve the last pan limits
        self.dragging_3d = False
        if event.button == 3:
            self.panning_pcd = False
            # now lock in the current axes limits so draw() won’t reset them
            self._zpcd_xlim = self.ax_pcd.get_xlim()
            self._zpcd_ylim = self.ax_pcd.get_ylim()
            self.zoom_pcd = True


    def _on_scroll(self, event):
        factor = 1.2 if event.button == 'up' else 1 / 1.2
        if event.inaxes == self.ax_pcd:
            self.zoom_pcd = True
            x0, x1 = self.ax_pcd.get_xlim()
            y0, y1 = self.ax_pcd.get_ylim()
            cx, cy = event.xdata, event.ydata
            self._zpcd_xlim = [cx - (cx - x0) * factor, cx + (x1 - cx) * factor]
            self._zpcd_ylim = [cy - (cy - y0) * factor, cy + (y1 - cy) * factor]
        elif event.inaxes == self.ax_img:
            self.zoom_img = True
            x0, x1 = self.ax_img.get_xlim()
            y0, y1 = self.ax_img.get_ylim()
            cx, cy = event.xdata, event.ydata
            self._zimg_xlim = [cx - (cx - x0) * factor, cx + (x1 - cx) * factor]
            self._zimg_ylim = [cy - (cy - y0) * factor, cy + (y1 - cy) * factor]
        self._draw()

    def _on_match(self):
        # preserve the current point-cloud view so it doesn't jump back
        self._zpcd_xlim = self.ax_pcd.get_xlim()
        self._zpcd_ylim = self.ax_pcd.get_ylim()
        self.zoom_pcd = True

        if self.selected_3d is None or self.selected_2d is None:
            messagebox.showwarning(
                "Selection required",
                "You must select both a 3D cone and a 2D cone to match."
            )
            return

        slam = self.slam_entries[self.selected_3d]
        yolo = self.yolo_entries[self.selected_2d]

        # --- conflict resolution ---
        final_type = slam['type']
        if slam['type'] != yolo['type']:
            prompt = (
                f"Type mismatch:\n"
                f" • 3D (SLAM) cone: {slam['type']}\n"
                f" • 2D (YOLO) cone: {yolo['type']}\n\n"
                "Enter final type (yellow_cone, blue_cone, orange_cone, big_cone):"
            )
            choice = simpledialog.askstring("Resolve conflict", prompt, parent=self.root)
            if not choice:
                return
            final_type = choice

        # --- build merged entry ---
        merged = {
            'id':    max(slam['id'], yolo['id']),
            'type':  final_type,
            'h':     slam['h'],  'w': slam['w'],  'l': slam['l'],
            'x':     slam['x'],  'y': slam['y'],  'z': slam['z'],
            'left':  yolo['left'], 'top': yolo['top'],
            'right': yolo['right'],'bottom': yolo['bottom']
        }

        # record for undo
        self.match_history.append({
            'slam_idx':   self.selected_3d,
            'slam_entry': slam,
            'yolo_idx':   self.selected_2d,
            'yolo_entry': yolo,
            'merged':     merged
        })

        # remove them from the live lists
        del self.slam_entries[self.selected_3d]
        del self.yolo_entries[self.selected_2d]

        # queue for writeout
        self.matched_entries.append(merged)

        print("Successfully matched!")
        self.selected_3d = None
        self.selected_2d = None
        self._draw()

    def _on_undo(self):
        """Undo the last match operation."""
        if not self.match_history:
            messagebox.showinfo("Undo", "Nothing to undo.")
            return

        last = self.match_history.pop()
        # remove the merged entry
        self.matched_entries.remove(last['merged'])
        # re-insert the original entries
        self.slam_entries.insert(last['slam_idx'], last['slam_entry'])
        self.yolo_entries.insert(last['yolo_idx'], last['yolo_entry'])
        print("Undid last match.")
        self._draw()

    def _on_reset(self):
        """Reset this frame back to its original state."""
        if messagebox.askyesno("Reset Frame",
                               "Discard all matches and changes for this frame?"):
            # clear matches & history
            self.match_history.clear()
            self.matched_entries.clear()
            # reload from disk
            self._load_frame()
            self.selected_3d = None
            self.selected_2d = None
            print("Frame reset.")
            self._draw()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="Dataset folder: labels/, images/, points/")
    args = p.parse_args()
    FrameLabeler(args.folder)
