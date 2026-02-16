# Gaze-Tracker

**Gaze-Tracker** is a Python-based tool for analyzing eye-tracking data and mapping gaze coordinates onto the plane of a visual object. The system takes eye-tracking recordings from wearable devices (e.g., Pupil Labs glasses) and produces gaze trajectories and heatmaps for visual analysis.

## Features

* Converts gaze coordinates from video recordings to object coordinates.
* Generates gaze trajectories over a target object.
* Produces dwell-time heatmaps on objects.
* Supports configurable parameters for keypoint detection, robust matching, and video processing.

## Input Data

Gaze-Tracker uses recordings from [**Pupil Player**](https://github.com/pupil-labs/pupil). Required files:

* `world.mp4` – video from the external camera
* `world.intrinsics` – camera intrinsics
* `world_timestamps.npy` – timestamps for each video frame
* `gaze.pldata` – gaze direction data

## Installation

1. Clone the repository:

```bash
git clone https://github.com/xvalex2/gaze-tracker.git
cd gaze-tracker
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install:

```bash
pip install .
```

## Scripts

### `gaze-tracker`

Converts gaze coordinates from video frames to the object plane and produces CSV and video outputs.

```bash
usage: gaze-tracker [-h] [options] data_path object video_out data_out
```

**Positional arguments:**

* `data_path` – path to the data folder
* `object` – image of the target object
* `video_out` – output video file
* `data_out` – output CSV file

**Options include:**

* `--start_frame`, `--end_frame` – process a subset of frames
* `--video_scale` – scale factor for output video
* `--undistort_alpha` – camera undistortion scaling
* `--lowe_filter_ratio`, `--min_matches` – feature matching parameters
* `--ransac`, `--rho`, `--lemeds` – outlier detection algorithms
* `--sift_contrast_threshold`, `--sift_edge_threshold` – SIFT parameters
* `--show_keypoints`, `--show_object`, `--show_object_keypoints` – visualization options

### `trajectory`

Draws gaze trajectories in the object plane from the CSV output of `gaze-tracker`.

```bash
usage: trajectory [-h] [--disable-confidence-filter] object gaze out_image
```

**Arguments:**

* `object` – image of the tracked object
* `gaze` – gaze CSV file in object coordinates
* `out_image` – output image file

**Option:**

* `--disable-confidence-filter` – include all gaze points, even those with zero confidence

### `heatmap`

Generates dwell-time heatmaps of gaze on the object plane.

```bash
usage: heatmap [-h] [--alpha ALPHA] [--sigma SIGMA] [--disable-confidence-filter] object gaze out_image
```

**Arguments:**

* `object` – image of the tracked object
* `gaze` – gaze CSV file in object coordinates
* `out_image` – output image file

**Options:**

* `--alpha` – transparency of heatmap overlay (default 0.5)
* `--sigma` – Gaussian blur sigma (default 10)
* `--disable-confidence-filter` – include all gaze points

---

## Example Workflow

```bash
# Step 1: Convert gaze data to object coordinates
gaze-tracker path/to/data painting.jpg output_video.mp4 gaze.csv

# Step 2: Generate gaze trajectory
trajectory painting.jpg gaze.csv trajectory.png

# Step 3: Generate heatmap
heatmap painting.jpg gaze.csv heatmap.png
```

## License

This project is licensed under the MIT License.
