# 🏎️ FSKITTI: Multi-Modal 3D Object Detection Dataset for Formula Student

**FSKITTI** is a multi-modal 3D object detection dataset tailored for the perception challenges of autonomous Formula Student race cars. It follows the official KITTI format and includes LiDAR point clouds, RGB camera images, 3D bounding boxes, and calibration files. The dataset was created from synchronized ROS bags collected by FST Lisboa’s autonomous system and is intended to support and benchmark both monocular and sensor-fusion-based perception pipelines.

---

## 📦 Dataset Overview

- **Name**: FSKITTI
- **Format**: KITTI (Multi-modal)
- **Sensors**:
  - 🎥 Monocular Camera (Lucid Triton TRI032S)
  - 🔦 40-beam Hesai Pandar40P LiDAR
- **Annotations**: 3D bounding boxes of colored traffic cones (orange, blue, yellow)
- **Use Cases**: LiDAR-only 3D detection, monocular 3D detection, multi-modal fusion
- **Scenes**: Multiple outdoor and indoor sessions from Formula Student testing sessions

---

## 📁 Directory Structure

Each scene is organized in the standard KITTI structure:

```
FSKITTI/
├── training/
│   ├── image_2/         # Monocular RGB images
│   ├── label_2/         # 3D object annotations (KITTI format)
│   ├── calib/           # Camera and LiDAR calibration files
│   └── velodyne/        # LiDAR point clouds in .bin format
└── testing/
    ├── image_2/         # Monocular RGB images
    ├── calib/           # Camera and LiDAR calibration files
    └── velodyne/        # LiDAR point clouds in .bin format
```

The calibration files include `P2`, `Tr_velo_to_cam`, and `R0_rect` matrices for compatibility with KITTI-based toolchains.

---

## 🧠 Motivation and Objectives

The primary goal of FSKITTI is to facilitate training and evaluation of 3D object detection models in environments typical of Formula Student. Unlike urban datasets such as KITTI, nuScenes, or Waymo, Formula Student scenarios involve simpler objects (cones), tighter spatial constraints, and specific domain challenges.

FSKITTI was designed to:
- Serve as a **benchmark** for LiDAR-based and camera-based cone detection.
- Enable **sensor fusion** experiments through accurate calibration and temporal synchronization.
- Provide a **lightweight, real-time-compatible** dataset for racing scenarios.

---

## 🛠️ Calibration and Labeling

- **Camera-LiDAR calibration** was performed using MATLAB-based checkerboard techniques to obtain accurate extrinsic and intrinsic parameters.
- All annotations are provided in the KITTI label format, with support for:
  - 3D object dimensions and orientation
  - Cone positions in the LiDAR frame
  - Transformation consistency across all data modalities

---

## 💡 Usage and Integration

### 📚 Training Models

This dataset is compatible with:
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- Any custom pipeline expecting KITTI-format input

### 🧪 Suggested Tasks

- 3D object detection from LiDAR-only input
- Monocular 3D cone detection
- Sensor fusion: camera + LiDAR
- Dataset augmentation and domain adaptation

---

## 📊 Sample Classes and Annotation Format

Classes include:

- `orange_cone`
- `blue_cone`
- `yellow_cone`

Each `label_2` entry contains:
```
<Class> <truncated> <occluded> <alpha> <bbox> <dimensions (h, w, l)> <location (x, y, z)> <rotation_y>
```
Units follow KITTI standard (meters for spatial dimensions).

---

## 🔍 Visualizations

Use tools such as:
- `kitti2bag` or `bag2kitti` for conversion
- Open3D or RViz for point cloud visualization
- MMDetection3D demo tools for viewing predictions

---

## 📜 License

To be defined. We recommend [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) for academic use. Please create a `LICENSE` file to formalize this.

---

## 📝 Citation

If you use this dataset in your research or experiments, please cite the following work:

```
@mastersthesis{valverde2025fskitti,
  title     = {3D Object Detection for Formula Student Autonomous Cars: Dataset and Benchmarks},
  author    = {Miguel Valverde},
  school    = {Instituto Superior Técnico},
  year      = {2025},
  note      = {Master's Thesis}
}
```

---

## 👨‍💼 Contact

Questions, issues, or suggestions? Feel free to open an issue or contact:

- Miguel Valverde – [miguel.heitor.valverde@tecnico.ulisboa.pt]

---

## 🔗 Related Projects

- [ConeScenes](https://github.com/Chalmers-Formula-Student/coneScenes)
- [FSOCO](https://github.com/pinakinathc/fscoco)
- [KITTI Vision Benchmark](http://www.cvlibs.net/datasets/kitti/)
