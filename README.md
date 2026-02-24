# Gaze-Tracker

**Gaze-Tracker** is a Python-based tool for analyzing eye-tracking data and mapping gaze coordinates onto the plane of a visual object. The system takes eye-tracking recordings from wearable devices (e.g., Pupil Labs glasses) and produces gaze trajectories and heatmaps for visual analysis.

## Features

* Converts gaze coordinates from video recordings to object coordinates.
* Generates gaze trajectories over a target object.
* Produces dwell-time heatmaps on objects.
* Supports configurable parameters for keypoint detection, robust matching, and video processing.

## Input Data

Gaze-Tracker uses recordings from [**Pupil Player**](https://github.com/pupil-labs/pupil). Required files:

* `world.mp4` - video from the external camera
* `world.intrinsics` - camera intrinsics
* `world_timestamps.npy` - timestamps for each video frame
* `gaze.pldata` - gaze data

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

## gaze-tracker

Transforms gaze coordinates from the video image plane (egocentric scene camera) into the reference object plane using feature-based homography estimation. The tool expects a data directory containing the eye-tracking video and gaze coordinates, and a reference image of the tracked object.

```
usage: gaze-tracker [-h] [--start_frame START_FRAME] [--end_frame END_FRAME]
                    [--video_scale VIDEO_SCALE]
                    [--undistort_alpha UNDISTORT_ALPHA]
                    [--disable_autocorrect]
                    [--lowe_filter_ratio LOWE_FILTER_RATIO]
                    [--min_matches MIN_MATCHES]
                    [--ste_threshold STE_THRESHOLD] [--gms_filter]
                    [--gms_threshold GMS_THRESHOLD]
                    [--robust_threshold ROBUST_THRESHOLD]
                    [--ransac | --rho | --lmeds]
                    [--sift_contrast_threshold SIFT_CONTRAST_THRESHOLD]
                    [--sift_edge_threshold SIFT_EDGE_THRESHOLD]
                    [--tracking_log TRACKING_LOG]
                    [--show_inliers] [--show_outliers]
                    [--show_keypoints] [--show_object]
                    [--show_matches]
                    data_path object video_out data_out
```

### Positional arguments

* **`data_path`** Path to the input data directory (video and gaze data).
* **`object`** Reference image of the planar object to be tracked.
* **`video_out`** Path to the output video file with visualization overlays.
* **`data_out`** Path to the output CSV file containing transformed gaze coordinates in the object coordinate system.

### Frame range and preprocessing

* **`--start_frame`** Index of the first frame to process.
* **`--end_frame`** Index of the last frame to process.
* **`--video_scale`** Scale factor applied to the output video resolution. Default: `1.2`.
* **`--undistort_alpha`** Undistortion scaling parameter.  
  `0` — only valid pixels are preserved  
  `1` — all original pixels are retained  
  Default: `0.5`.
* **`--disable_autocorrect`** Disable automatic brightness and contrast correction.

### Feature detection and matching

* **`--sift_contrast_threshold`** SIFT contrast threshold. Higher values reduce the number of detected features. Default: `0.04`.
* **`--sift_edge_threshold`** SIFT edge threshold. Higher values retain more edge-like features. Default: `10.0`.
* **`--lowe_filter_ratio`** Lowe ratio test threshold for KNN matching.  
  `0` — reject all matches  
  `1` — no ratio filtering  
  Default: `0.8`.
* **`--gms_filter`** Enable Grid-based Motion Statistics (GMS) filtering.
* **`--gms_threshold`** GMS consistency threshold. Larger values enforce stricter filtering. Default: `3`.
* **`--min_matches`** Minimum number of matches required to estimate homography. Default: `20`.

### Robust homography estimation

One robust estimator can be selected:

* **`--lmeds`** Use LMedS for outlier detection and automatically fall back to RANSAC if the inlier ratio is below 50%. This is the default.
* **`--ransac`** Use RANSAC for outlier detection.
* **`--rho`** Use RHO-based robust estimation.

Additional parameters:

* **`--robust_threshold`** Reprojection error threshold (in pixels) used by RANSAC or RHO. Default: `5`.
* **`--ste_threshold`** Maximum allowed symmetric transfer error (in pixels) for accepting a homography. Default: `50`.

### Logging and visualization

* **`--tracking_log`** Output file for detailed tracking diagnostics.
* **`--show_inliers`** Visualize inlier keypoints.
* **`--show_outliers`** Visualize rejected keypoints.
* **`--show_keypoints`** Show all detected keypoints, including unmatched ones.
* **`--show_object`** Display the reference object and overlay the transformed gaze trajectory.
* **`--show_matches`** Visualize keypoint correspondences between the frame and the reference object.

## trajectory

Draws gaze trajectories in the object plane from the CSV output of `gaze-tracker`.

```bash
usage: trajectory [-h] [--color COLOR] [--thickness THICKNESS] [--disable-confidence-filter] object gaze out_image
```

**Arguments:**

* `object` Image of the tracked object
* `gaze` Gaze CSV file in object coordinates
* `out_image` Output image file

**Options:**

* `--disable-confidence-filter` Include all gaze points, even those with zero confidence.
* `--color` Trajectory color in #RRGGBB format. Default: black.
* `--thickness` Trajectory thicknesst. Default: `1`.

## heatmap

Generates dwell-time heatmaps of gaze on the object plane.

```bash
usage: heatmap [-h] [--alpha ALPHA] [--sigma SIGMA] [--disable-confidence-filter] object gaze out_image
```

**Arguments:**

* `object` Image of the tracked object
* `gaze` Gaze CSV file in object coordinates
* `out_image` Output image file

**Options:**

* `--alpha` Transparency of heatmap overlay (default 0.5)
* `--sigma` Gaussian blur sigma (default 10)
* `--disable-confidence-filter` Include all gaze points, even those with zero confidence

## Example Workflow

```bash
# Activate a virtual environment
source gaze-tracker/venv/bin/activate       # Linux
call gaze-tracker\venv\Scripts\activate.bat # Windows

# Step 1: Convert gaze data to object coordinates
gaze-tracker path/to/data painting.jpg output_video.mp4 gaze.csv

# Step 2: Generate gaze trajectory
trajectory painting.jpg gaze.csv trajectory.png

# Step 3: Generate heatmap
heatmap painting.jpg gaze.csv heatmap.png
```

## License

This project is licensed under the MIT License.
