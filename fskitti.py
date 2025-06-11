import os
import shutil
import random
import numpy as np

# Configuration
BASE_DIR = './fskitti'
DEST_DIR = '../bags/dataset8'
SEED = 42
TRAIN_RATIO = 0.8

SELECTED_SCENES = [
    'camera_alverca_autox_april2',
    'camera_alverca_autox_may1',
    'camera_central_noise_rain',
    'camera_estoril_autox1',
    'camera_estoril_autox2',
   # 'camera_thesis_alverca_skidpad'
]

# Sensor parameters
intrinsic_flat = [1801.762859401526, 0.0, 1012.815860372736,
                  0.0, 1800.131336129669, 716.053488689367,
                  0.0, 0.0, 1.0]

# Keep this for calib files
extrinsic_flat = [0.0272532712, -0.9996134758, -0.0054916535, -0.0421333031,
                  0.0050777668,  0.0056320583, -0.9999712477, -0.0401744490,
                  0.9996156639,  0.0272246023,  0.0052292962, 0.1868387551]

ext_matrix_rot = [-0.028513661350487, -0.999372268994208, 0.021024725485918, 0.112705030407873, 
                    -0.003409584841331, -0.020935917420255, -0.999775005735281, -0.056709728245902,
                    0.999587587881922,  -0.028578931525527,  -0.002810484879827 , 0.006342076987547]

# Calibration transformation matrix: LiDAR → Camera
Tr_velo_to_cam_flat = [-0.028513661350487, -0.999372268994208, 0.021024725485918, 0.112705030407873, 
                    -0.003409584841331, -0.020935917420255, -0.999775005735281, -0.056709728245902,
                    0.999587587881922,  -0.028578931525527,  -0.002810484879827 , 0.006342076987547]

# Decompose into R and t
R_velo_to_cam = np.array(Tr_velo_to_cam_flat).reshape(3, 4)[:, :3]
t_velo_to_cam = np.array(Tr_velo_to_cam_flat).reshape(3, 4)[:, 3]

# Prepare calibration text block
def generate_calib_file(intrinsic_flat, extrinsic_flat):
    fmt = lambda row, t: ' '.join(f"{v:.12e}" for v in row) + f" {t:.12e}"
    K = [intrinsic_flat[0:3], intrinsic_flat[3:6], intrinsic_flat[6:9]]
    P_lines = [f"P{i}: " + fmt(K[0], 0.0) + ' ' + fmt(K[1], 0.0) + ' ' + fmt(K[2], 0.0) for i in range(4)]
    R0 = "R0_rect: " + ' '.join(f"{v:.12e}" for v in [1,0,0, 0,1,0, 0,0,1])
    Tr = "Tr_velo_to_cam: " + ' '.join(f"{v:.12e}" for v in extrinsic_flat)
    Tr_imu = "Tr_imu_to_velo: " + ' '.join(f"{v:.12e}" for v in [1,0,0,0, 0,1,0,0, 0,0,1,0])
    return '\n'.join(P_lines + [R0, Tr, Tr_imu]) + '\n'

CALIB_CONTENT = generate_calib_file(intrinsic_flat, extrinsic_flat)

# Prepare directories
splits = {
    'training': ['image_2', 'label_2', 'velodyne', 'calib'],
    'testing':  ['image_2',             'velodyne', 'calib'],
}
for split, subdirs in splits.items():
    for sub in subdirs:
        os.makedirs(os.path.join(DEST_DIR, split, sub), exist_ok=True)

imagesets_dir = os.path.join(DEST_DIR, 'ImageSets')
os.makedirs(imagesets_dir, exist_ok=True)

# Check scene existence
all_scenes = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))])
invalid = [s for s in SELECTED_SCENES if s not in all_scenes]
if invalid:
    raise ValueError(f"Selected scenes not found: {invalid}")
scenes = SELECTED_SCENES

# Collect frames
global_frames = []
print("Frame counts per scene:")
for scene in scenes:
    path = os.path.join(BASE_DIR, scene)
    imgs = sorted(os.listdir(os.path.join(path, 'images')))
    lbls = sorted(os.listdir(os.path.join(path, 'labels')))
    pcods = sorted(os.listdir(os.path.join(path, 'points')))
    if not (len(imgs) == len(lbls) == len(pcods)):
        raise RuntimeError(f"Mismatch in scene {scene}")
    print(f"  {scene}: {len(imgs)}")
    for img, lbl, pcd in zip(imgs, lbls, pcods):
        global_frames.append((path, img, lbl, pcd))

# Split train/test
total = len(global_frames)
train_n = int(total * TRAIN_RATIO)
random.seed(SEED)
indices = list(range(total))
random.shuffle(indices)
train_idx = set(indices[:train_n])
train_frames = [global_frames[i] for i in indices if i in train_idx]
test_frames  = [global_frames[i] for i in indices if i not in train_idx]
print(f"Total={total}, train={len(train_frames)}, test={len(test_frames)}")

# Transformation function for label coordinates
def transform_label_line_to_camera(line):
    parts = line.strip().split()
    if len(parts) < 15:
        return line
    pos_lidar = np.array([float(parts[11]), float(parts[12]), float(parts[13])])
    pos_cam = R_velo_to_cam @ pos_lidar + t_velo_to_cam
    parts[11], parts[12], parts[13] = [f"{x:.6f}" for x in pos_cam]
    return ' '.join(parts)

# Copy function
def copy_frames(frames, split):
    for i, (scene, img, lbl, pcd) in enumerate(frames):
        nid = f"{i:06d}"
        shutil.copy(os.path.join(scene, 'images', img), os.path.join(DEST_DIR, split, 'image_2', nid + '.png'))
        if split == 'training':
            with open(os.path.join(scene, 'labels', lbl), 'r') as f_in:
                transformed_lines = [transform_label_line_to_camera(line) for line in f_in]
            with open(os.path.join(DEST_DIR, split, 'label_2', nid + '.txt'), 'w') as f_out:
                f_out.write('\n'.join(transformed_lines) + '\n')
        shutil.copy(os.path.join(scene, 'points', pcd), os.path.join(DEST_DIR, split, 'velodyne', nid + '.bin'))
        with open(os.path.join(DEST_DIR, split, 'calib', nid + '.txt'), 'w') as f:
            f.write(CALIB_CONTENT)

# Export
copy_frames(train_frames, 'training')
copy_frames(test_frames, 'testing')

# ImageSets
train_ids = [f"{i:06d}" for i in range(len(train_frames))]
test_ids  = [f"{i:06d}" for i in range(len(test_frames))]
for nm, ids in [('train', train_ids), ('val', train_ids), ('trainval', train_ids), ('test', test_ids)]:
    with open(os.path.join(imagesets_dir, f"{nm}.txt"), 'w') as f:
        f.write("\n".join(ids))

print("✅ Done: exported dataset with 3D boxes converted from camera to LiDAR frame.")
