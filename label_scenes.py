#!/usr/bin/env python3
import os
import glob
import numpy as np
import tkinter as tk
import cv2
from tkinter import messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

ORIG_W, ORIG_H = 2048, 1536

def parse_label_file(path):
    slam_entries, yolo_entries = [], []
    with open(path, 'r') as f:
        for idx, ln in enumerate(f):
            parts = ln.strip().split()
            if len(parts) < 14: continue
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
    def is_empty(e):
        return all(e.get(k,0.0)==0.0 for k in
                   ('left','top','right','bottom','h','w','l','x','y','z'))
    lines = []
    # SLAM-only
    for e in slam_entries:
        if is_empty(e): continue
        lines.append(
            f"{e['type']} 0.00 0 0.00 "
            f"0.00 0.00 0.00 0.00 "
            f"{e['h']:.3f} {e['w']:.3f} {e['l']:.3f} "
            f"{e['x']:.3f} {e['y']:.3f} {e['z']:.3f} 0.00"
        )
    # Matched
    for e in matched_entries:
        lines.append(
            f"{e['type']} 0.00 0 0.00 "
            f"{e['left']:.2f} {e['top']:.2f} {e['right']:.2f} {e['bottom']:.2f} "
            f"{e['h']:.3f} {e['w']:.3f} {e['l']:.3f} "
            f"{e['x']:.3f} {e['y']:.3f} {e['z']:.3f} 0.00"
        )
    # YOLO-only
    for e in yolo_entries:
        if is_empty(e): continue
        lines.append(
            f"{e['type']} 0.00 0 0.00 "
            f"{e['left']:.2f} {e['top']:.2f} {e['right']:.2f} {e['bottom']:.2f} "
            f"0.00 0.00 0.00 0.00 0.00 0.00 0.00"
        )
    with open(path,'w') as f:
        f.write('\n'.join(lines))

class FrameLabeler:
    def __init__(self, folder):
        self.folder = folder
        self.frame_ids = sorted(os.path.splitext(os.path.basename(p))[0]
                                for p in glob.glob(os.path.join(folder,'labels','*.txt')))
        self.idx = 0

        # interaction state
        self.selected_3d = None
        self.selected_2d = None
        self.dragging_3d = False
        self.dragging_2d = False
        self.creation_start = None
        self.zoom_pcd = False
        self.zoom_img = False
        self.panning_pcd = False

        self.match_history = []
        self.matched_entries = []

        self._load_frame()

        # build UI
        self.root = tk.Tk()
        self.root.title("Label Editor")

        # ── Canvas ───────────────────────────────────────────────────────────
        self.fig = Figure(figsize=(12,6))
        self.ax_pcd = self.fig.add_subplot(121)
        self.ax_img = self.fig.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, self.root).update()

        # connect events
        c = self.canvas.mpl_connect
        c("button_press_event",   self._on_click)
        c("motion_notify_event",  self._on_drag)
        c("button_release_event", self._on_release)
        c("scroll_event",         self._on_scroll)

        self._draw()

        # ── Bottom toolbar (centered) ────────────────────────────────────────
        bottom = tk.Frame(self.root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, pady=4)

        # left spacer
        tk.Frame(bottom).pack(side=tk.LEFT, expand=True)

        for text, cmd in [
            ("Save + Next", self._on_next),
            ("Match Cones", self._on_match),
            ("Undo Last",   self._on_undo),
            ("Reset Frame", self._on_reset),
            ("Delete Cone", self._on_delete),
        ]:
            tk.Button(bottom, text=text, command=cmd).pack(side=tk.LEFT, padx=8)

        # right spacer
        tk.Frame(bottom).pack(side=tk.LEFT, expand=True)

        tk.mainloop()

    def _load_frame(self):
        fid = self.frame_ids[self.idx]
        pts = np.fromfile(os.path.join(self.folder,'points',f"{fid}.bin"),
                          dtype=np.float32).reshape(-1,5)
        self.pcd = pts[:,:3]
        img = cv2.imread(os.path.join(self.folder,'images',f"{fid}.png"))
        if img is None: raise FileNotFoundError(f"{fid}.png missing")
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = os.path.join(self.folder,'labels',f"{fid}.txt")
        self.slam_entries, self.yolo_entries = parse_label_file(lab)

        # next‐ID for any new box
        all_ids = [e['id'] for e in self.slam_entries + self.yolo_entries]
        self.next_id = max(all_ids, default=-1) + 1

    def _get_color(self,typ):
        return {
            'yellow_cone':'yellow','blue_cone':'blue',
            'orange_cone':'orange','big_cone':'purple'
        }.get(typ,'gray')
    def _get_bgr(self,typ):
        return {
            'yellow_cone':(255,255,0),
            'blue_cone':  (0,0,255),
            'orange_cone':(255,165,0),
            'big_cone':   (255,0,255)
        }.get(typ,(192,192,192))

    def _draw(self):
        # --- PCD pane ---
        self.ax_pcd.clear()
        rx = -self.pcd[:,1]; ry = self.pcd[:,0]
        self.ax_pcd.scatter(rx, ry, s=1)
        fov, R = np.radians(32.5), 40
        for a in [-fov,fov]:
            ex,ey = R*np.cos(a),R*np.sin(a)
            self.ax_pcd.plot([-ey,  -ey+0],[ ex, ex+0], color='red',alpha=0.3)
        for i,e in enumerate(self.slam_entries):
            if i==self.selected_3d: continue
            x,y,l,w = e['x'],e['y'],e['l'],e['w']
            rpx, rpy = -y, x
            rect = plt.Rectangle((rpx-w/2,rpy-l/2),w,l,fill=False,
                                 edgecolor=self._get_color(e['type']),linewidth=2)
            self.ax_pcd.add_patch(rect)
        if self.selected_3d is not None:
            e = self.slam_entries[self.selected_3d]
            rpx,rpy,w,l = -e['y'], e['x'], e['w'], e['l']
            self.ax_pcd.add_patch(
                plt.Rectangle((rpx-w/2,rpy-l/2),w,l,fill=False,
                              edgecolor='red',linewidth=3))
        self.ax_pcd.set_xlabel('X'); self.ax_pcd.set_ylabel('Y'); self.ax_pcd.axis('equal')
        if self.zoom_pcd:
            self.ax_pcd.set_xlim(self._zpcd_xlim); self.ax_pcd.set_ylim(self._zpcd_ylim)
        else:
            self.ax_pcd.set_xlim(-30,30); self.ax_pcd.set_ylim(0,30)

        # --- Image pane ---
        self.ax_img.clear()
        h,w = self.img.shape[:2]
        sx,sy = w/ORIG_W, h/ORIG_H
        overlay = self.img.copy()
        for i,e in enumerate(self.yolo_entries):
            if i==self.selected_2d: continue
            x1,y1 = int(e['left']*sx), int(e['top']*sy)
            x2,y2 = int(e['right']*sx),int(e['bottom']*sy)
            cv2.rectangle(overlay,(x1,y1),(x2,y2),self._get_bgr(e['type']),2)
        if self.selected_2d is not None:
            e = self.yolo_entries[self.selected_2d]
            x1,y1 = int(e['left']*sx), int(e['top']*sy)
            x2,y2 = int(e['right']*sx),int(e['bottom']*sy)
            cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,0,0),3)
        self.ax_img.imshow(overlay); self.ax_img.axis('off')
        if self.zoom_img:
            self.ax_img.set_xlim(self._zimg_xlim); self.ax_img.set_ylim(self._zimg_ylim)

        self.canvas.draw()

    def _on_click(self, event):
        # --- right‐click pan ---
        if event.button==3 and event.inaxes==self.ax_pcd and event.xdata:
            self.panning_pcd=True
            self.pan_start=(event.xdata,event.ydata)
            self.pan_xlim0=self.ax_pcd.get_xlim()
            self.pan_ylim0=self.ax_pcd.get_ylim()
            return

        # --- PCD selection/drag ---
        if event.inaxes==self.ax_pcd and event.xdata:
            click_x, click_y = event.ydata, -event.xdata
            dists = [np.hypot(click_x-e['x'],click_y-e['y'])
                     for e in self.slam_entries]
            idx = int(np.argmin(dists))
            if dists[idx]<1.0:
                self.selected_3d=idx
                self.dragging_3d=True
            else:
                self.selected_3d=None
            self._draw()
            return

        # --- Image pane: creation vs selection vs start‐drag ---
        if event.inaxes==self.ax_img and event.xdata:
            x,y=event.xdata,event.ydata
            h,w=self.img.shape[:2]; sx,sy=w/ORIG_W,h/ORIG_H

            # if we're in the middle of drawing a new box
            if self.creation_start is None:
                # see if clicked in existing box
                for i,e in enumerate(self.yolo_entries):
                    x1,y1 = e['left']*sx, e['top']*sy
                    x2,y2 = e['right']*sx, e['bottom']*sy
                    if x1<=x<=x2 and y1<=y<=y2:
                        self.selected_2d=i
                        self.dragging_2d=True
                        self.drag_offset=(x-e['left']*sx, y-e['top']*sy)
                        break
                else:
                    # start a new box
                    self.creation_start = (x,y)
                    self.selected_2d=None
                self._draw()
            else:
                # finish new‐box creation
                x0, y0 = self.creation_start
                x1, y1 = event.xdata, event.ydata
                # convert pixel → original coords
                left, right = sorted([x0 / sx, x1 / sx])
                top, bottom = sorted([y0 / sy, y1 / sy])

                typ = simpledialog.askstring(
                    "New box type",
                    "Enter cone type (yellow_cone, blue_cone, orange_cone, big_cone):",
                    parent=self.root
                )
                if typ:
                    new = {
                        'id':     self.next_id,
                        'type':   typ,
                        'left':   left,
                        'top':    top,
                        'right':  right,
                        'bottom': bottom
                    }
                    self.next_id += 1
                    self.yolo_entries.append(new)

                self.creation_start = None
                self._draw()

    def _on_drag(self, event):
        # panning
        if self.panning_pcd and event.inaxes==self.ax_pcd and event.xdata:
            dx,dy = self.pan_start[0]-event.xdata, self.pan_start[1]-event.ydata
            x0,x1 = self.pan_xlim0; y0,y1=self.pan_ylim0
            self.ax_pcd.set_xlim(x0+dx, x1+dx); self.ax_pcd.set_ylim(y0+dy, y1+dy)
            self.canvas.draw()
            return

        # dragging 3D
        if self.dragging_3d and self.selected_3d is not None and event.inaxes==self.ax_pcd:
            self.slam_entries[self.selected_3d]['x']=event.ydata
            self.slam_entries[self.selected_3d]['y']=-event.xdata
            self._draw()
            return

        # dragging 2D
        if self.dragging_2d and self.selected_2d is not None and event.inaxes == self.ax_img:
            # original pixel → ORIG coords
            h, w = self.img.shape[:2]
            sx, sy = w / ORIG_W, h / ORIG_H
            e = self.yolo_entries[self.selected_2d]

            # store original dims
            ow = e['right'] - e['left']
            oh = e['bottom'] - e['top']

            # apply drag offset & convert back
            ex = event.xdata - self.drag_offset[0]
            ey = event.ydata - self.drag_offset[1]
            new_left  = ex / sx
            new_top   = ey / sy

            e['left']   = new_left
            e['top']    = new_top
            e['right']  = new_left + ow
            e['bottom'] = new_top  + oh

            self._draw()
            return

    def _on_release(self, event):
        self.dragging_3d=False
        self.dragging_2d=False
        if event.button==3:
            self.panning_pcd=False
            self._zpcd_xlim=self.ax_pcd.get_xlim()
            self._zpcd_ylim=self.ax_pcd.get_ylim()
            self.zoom_pcd=True

    def _on_scroll(self, event):
        factor = 1.2 if event.button=='up' else 1/1.2
        if event.inaxes==self.ax_pcd:
            self.zoom_pcd=True
            x0,x1=self.ax_pcd.get_xlim(); y0,y1=self.ax_pcd.get_ylim()
            cx,cy=event.xdata,event.ydata
            self._zpcd_xlim=[cx-(cx-x0)*factor, cx+(x1-cx)*factor]
            self._zpcd_ylim=[cy-(cy-y0)*factor, cy+(y1-cy)*factor]
        elif event.inaxes==self.ax_img:
            self.zoom_img=True
            x0,x1=self.ax_img.get_xlim(); y0,y1=self.ax_img.get_ylim()
            cx, cy = event.xdata, event.ydata
            self._zimg_xlim=[cx-(cx-x0)*factor, cx+(x1-cx)*factor]
            self._zimg_ylim=[cy-(cy-y0)*factor, cy+(y1-cy)*factor]
        self._draw()

    def _on_match(self):
        # preserve the current point-cloud view
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

        # conflict resolution
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

        # build merged entry
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
        if messagebox.askyesno("Reset","Discard all changes for this frame?"):
            self.match_history.clear(); self.matched_entries.clear()
            self._load_frame()
            self.selected_3d=self.selected_2d=None
            self._draw()

    def _on_delete(self):
        """Delete the currently selected cone, 3D or 2D."""
        if self.selected_3d is not None:
            del self.slam_entries[self.selected_3d]
            self.selected_3d = None
            self._draw()
        elif self.selected_2d is not None:
            del self.yolo_entries[self.selected_2d]
            self.selected_2d = None
            self._draw()
        else:
            messagebox.showinfo("Delete", "No cone selected to delete.")

    def _on_next(self):
        fid = self.frame_ids[self.idx]
        path = os.path.join(self.folder,'labels',f"{fid}.txt")
        write_label_file(path,
                         self.slam_entries,
                         self.yolo_entries,
                         self.matched_entries)
        self.matched_entries.clear(); self.match_history.clear()
        self.idx += 1
        if self.idx>=len(self.frame_ids):
            messagebox.showinfo("Done","All frames processed.")
            self.root.quit(); return
        self._load_frame()
        self.selected_3d=self.selected_2d=None
        self._draw()

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("folder", help="Dataset folder")
    args=p.parse_args()
    FrameLabeler(args.folder)