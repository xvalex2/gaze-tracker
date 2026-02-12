import numpy as np
import cv2
import argparse
import msgpack
from os.path import join
from os.path import isfile
from typing import NamedTuple
import time
import csv

DEFAULT_VIDEO_SCALE = 1.2
DEFAULT_UNDISTORT_ALPHA = 0.5
DEFAULT_LOWE_FILTER_RATIO = 0.8
DEFAULT_MIN_MATCHES = 20

DEFAULT_ROBUST_METHOD = cv2.RANSAC
DEFAULT_ROBUST_THRESHOLD = 5

DEFAULT_SIFT_CONTRAST_THRESHOLD = 0.04
DEFAULT_SIFT_EDGE_THRESHOLD = 10.0


def lowe_filter(matches, ratio):
    result = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            result.append(m)
    return result


def find_homography(frame_keypoints, object_keypoints, matches, method = DEFAULT_ROBUST_METHOD, threshold = DEFAULT_ROBUST_THRESHOLD):
    # Find homography
    src_pts = np.float32([ object_keypoints[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ frame_keypoints[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    h, mask = cv2.findHomography(src_pts, dst_pts, method, threshold)

    # Find outliers
    if mask is not None:
        mask = mask.ravel()
        outliers = [match for match, is_inlier in zip(matches, mask) if is_inlier == 0]

    return h, outliers


def draw_bounding_box(frame, homography, object_w, object_h, line_color, line_width):
    rect = np.float32([ [0, 0], [0, object_h], [object_w, object_h], [object_w, 0] ]).reshape(-1,1,2)
    rect = cv2.perspectiveTransform(rect, homography)
    return cv2.polylines(frame, [np.int32(rect)], True, line_color, line_width, cv2.LINE_AA)


def draw_text(image, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (128, 0, 0)
    line_spacing = 1.5

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    line_height = int(text_size[1] * line_spacing)
    for i, line in enumerate(text.split("\n")):
        cv2.putText(image, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y = y + line_height


def handle_events():
    key = cv2.pollKey()
    if key == -1:
        return False

    if key == 27: # 'Esc' key to stop
        return True

    if key & 0xFF == ord('q'): # 'q' key to stop
        return True

    if key & 0xFF == ord(' '):
        # space pressed (pause)
        while cv2.waitKey(1) & 0xFF != ord(' '):
            time.sleep(0.1)

    return False


def frame_generator(filename, start_frame = None, end_frame = None):
    """
    Frame generator for a video file using cv2.VideoCapture

    Example:
        for frame_idx, frame in frame_generator("video.mp4", 100, 200):
            do_something(frame)
    """
    cap = cv2.VideoCapture(filename)

    frame_idx = 0
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

    while (end_frame is None) or (frame_idx < end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_idx, frame
        frame_idx += 1

    cap.release()


class VideoProps(NamedTuple):
    """
    Properties of a video file

    Example:
        props = VideoProps.from_file("world.mp4")
        print(f"Video resolution: {props.width} x {props.height}")
        print(f"Number of frames: {props.frames")
        print(f"FPS: {props.fps}")
    """

    width: int
    height: int
    fps: float
    frames: int

    @classmethod
    def from_file(cls, filename):
        cap = cv2.VideoCapture(filename)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return cls(width, height, fps, frames)


class Intrinsics(NamedTuple):
    """
    Camera intrinsics.

    Example:
        video_props = VideoProps.from_file("world.mp4")
        resolution = video_props.width, video_props.height
        intrinsics = Intrinsics.from_file("world.intrinsics", resolution)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsics.camera_matrix, intrinsics.dist_coefs, resolution, alpha, resolution)
    """

    camera_matrix: np.ndarray
    dist_coefs: np.ndarray
    cam_type: str

    @classmethod
    def from_file(cls, filename, resolution):
        with open(filename, "rb") as f:
            intrinsics = msgpack.unpack(f)
        intrinsics = intrinsics[str(resolution)]
        return cls(np.array(intrinsics['camera_matrix']), np.array(intrinsics['dist_coefs']), intrinsics['cam_type'])


class Undistorter:
    def __init__(self, camera_matrix, dist_coefs, resolution, alpha, new_resolution = None):
        self.camera_matrix = camera_matrix
        self.dist_coefs = dist_coefs
        self.resolution = resolution
        self.new_resolution = new_resolution if new_resolution else resulution
        self.alpha = alpha
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, resolution, alpha, new_resolution)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coefs, None, self.new_camera_matrix, new_resolution, cv2.CV_32FC1)

    def undistort_image(self, image):
        return cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)

    def undistort_points(self, points):
        # https://stackoverflow.com/questions/62170402/opencv-undistort-for-images-and-undistortpoints-are-inconsistent
        return cv2.undistortPointsIter(points, self.camera_matrix, self.dist_coefs, None, self.new_camera_matrix, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.03))


class GazeData:
    def __init__(self, gaze_filename, timestamps_filename):
        gaze = []
        with open(gaze_filename, "rb") as fh:
            for topic, payload in msgpack.Unpacker(fh, use_list=False, strict_map_key=False):
                entry = msgpack.unpackb(payload, use_list=False, strict_map_key=False)
                gaze.append({
                    'timestamp':  entry["timestamp"],
                    'confidence': entry["confidence"],
                    'norm_pos_x': entry['norm_pos'][0],
                    'norm_pos_y': entry['norm_pos'][1],
                })
        self.gaze = sorted(gaze, key=lambda x: x['timestamp'])

        timestamps = np.load(timestamps_filename)

        iframe = 0
        igaze = 0
        self.gaze_range_for_frame = dict()
        while (iframe < len(timestamps)-1) and (igaze < len(gaze)):
            igaze_start = igaze
            t = timestamps[iframe+1]
            while ((igaze < len(self.gaze)) and (self.gaze[igaze]['timestamp'] < t)):
                self.gaze[igaze]['frame'] = iframe
                self.gaze[igaze]['frame_timestamp'] = timestamps[iframe]
                igaze += 1
            self.gaze_range_for_frame[iframe] = (igaze_start, igaze)
            iframe += 1

    def gaze_for_frame(self, frame):
        try:
            begin, end = self.gaze_range_for_frame[frame]
            return self.gaze[begin:end]
        except KeyError:
            return []


class DataWriter:
    def __init__(self, filename):
        self.csvfile = open(filename, 'w', newline='', encoding='utf-8')
        self.fieldnames = ['timestamp', 'confidence', 'norm_pos_x', 'norm_pos_y', 'frame', 'frame_timestamp', 'object_x', 'object_y']
        self.writer = csv.DictWriter(self.csvfile, fieldnames=self.fieldnames)
        self.writer.writeheader()

    def write(self, gaze):
        for entry in gaze:
            self.writer.writerow(entry)

    def close(self):
        self.csvfile.close()


class TrackedObject:
    def __init__(self, filename, detector):
        self.filename = filename
        self.image = cv2.imread(filename)
        self.h, self.w, _ = self.image.shape
        self.keypoints, self.descriptors = detector.detectAndCompute(self.image, None)


def main():
    parser = argparse.ArgumentParser(
        prog='gaze-tracker',
        description='Converts eye-tracking coordinates from video plane to the 2D plane of the observed object.')
    parser.add_argument('data_path', help='Path to the data folder')
    parser.add_argument('object',    help='Image of object to track')
    parser.add_argument('video_out', help='Resulting video file')
    parser.add_argument('data_out',  help='Resulting CSV data file')

    parser.add_argument('--start_frame', required=False, default=None, type=int,
        help='First frame to process.')
    parser.add_argument('--end_frame', required=False, default=None, type=int,
        help='Last frame to process.')

    parser.add_argument('--video_scale', required=False, default=DEFAULT_VIDEO_SCALE, type=float,
        help=f'Scale output video. Default is {DEFAULT_VIDEO_SCALE}.')
    parser.add_argument('--undistort_alpha', required=False, default=DEFAULT_UNDISTORT_ALPHA, type=float,
        help=f'Undistortion scaling parameter. 0 - all the pixels in the undistorted image are valid. 1 - all the source image pixels are retained in the undistorted image. Default is {DEFAULT_UNDISTORT_ALPHA}.')
    parser.add_argument('--lowe_filter_ratio', required=False, default=DEFAULT_LOWE_FILTER_RATIO, type=float,
        help=f'Lowe filter ratio. 0 - filter out everything. 1 - no filtering. Default is: {DEFAULT_LOWE_FILTER_RATIO}.')
    parser.add_argument('--min_matches', required=False, default=DEFAULT_MIN_MATCHES, type=int,
        help=f'Minimum number of matches for homography estimation. Default is: {DEFAULT_MIN_MATCHES}.')

    parser.add_argument('--robust_threshold', required=False, default=DEFAULT_ROBUST_THRESHOLD, type=float,
        help=f'Threshold for RANSAC/RHO robust method. Default: {DEFAULT_ROBUST_THRESHOLD}')
    robust_method_group = parser.add_mutually_exclusive_group()
    robust_method_group.add_argument('--ransac', action='store_const', dest='robust_method', const=cv2.RANSAC,
        help='Use RANSAC robust method')
    robust_method_group.add_argument('--rho', action='store_const', dest='robust_method', const=cv2.RHO,
        help='Use RHO robust method')
    robust_method_group.add_argument('--lemeds', action='store_const', dest='robust_method', const=cv2.LMEDS,
        help='Use LMedS robust method')

    parser.add_argument('--sift_contrast_threshold', required=False, default=DEFAULT_SIFT_CONTRAST_THRESHOLD, type=float,
        help=f'SIFT detector contrast threshold. The larger the threshold, the less features are produced by the detector. Default is {DEFAULT_SIFT_CONTRAST_THRESHOLD}.')
    parser.add_argument('--sift_edge_threshold', required=False, default=DEFAULT_SIFT_EDGE_THRESHOLD, type=float,
        help=f'SIFT detector edge threshold. The larger the threshold, the less features are filtered out. Default is {DEFAULT_SIFT_EDGE_THRESHOLD}.')
    
    parser.add_argument('--show_object', action='store_true', help='Show object')
    parser.add_argument('--show_object_keypoints', action='store_true', help='Show object')
    parser.add_argument('--show_keypoints', action='store_true', help='Show keypoints')

    parser.set_defaults(robust_method=DEFAULT_ROBUST_METHOD)
    args = parser.parse_args()

    world_filename = join(args.data_path, 'world.mp4')
    intrinsics_filename = join(args.data_path, 'world.intrinsics')
    timestamps_filename = join(args.data_path, 'world_timestamps.npy')
    gaze_filename = join(args.data_path, 'gaze.pldata')

    source_files = (world_filename, intrinsics_filename, timestamps_filename, gaze_filename, args.object)
    for f in source_files:
        if not isfile(f):
            print(f"Error: file does not extst: {f}")
            exit(-1)

    # Input video, intrinsics and undistorter
    video_props = VideoProps.from_file(world_filename)
    resolution = video_props.width, video_props.height
    out_resolution = int(video_props.width * args.video_scale), int(video_props.height * args.video_scale)
    intrinsics = Intrinsics.from_file(intrinsics_filename, resolution)
    undistorter = Undistorter(intrinsics.camera_matrix, intrinsics.dist_coefs, resolution, args.undistort_alpha, out_resolution)

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(args.video_out, fourcc, video_props.fps, out_resolution)

    # Output data
    data_writer = DataWriter(args.data_out)

    # Gaze data
    gaze_data = GazeData(gaze_filename, timestamps_filename)

    # Detector and matcher
    detector = cv2.SIFT_create(contrastThreshold=args.sift_contrast_threshold, edgeThreshold=args.sift_edge_threshold)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    # Object to track
    obj = TrackedObject(args.object, detector)

    for frame_idx, frame in frame_generator(world_filename, args.start_frame, args.end_frame):
        h, outliers, num_outliers, num_inliers = None, None, 0, 0
        object_image = obj.image.copy()

        # Undistort
        out_frame = undistorter.undistort_image(frame)

        # Find keypoints
        keypoints, descriptors = detector.detectAndCompute(out_frame, None)

        # Match & filter matches
        matches = matcher.knnMatch(descriptors, obj.descriptors, k = 2)
        filtered_matches = lowe_filter(matches, args.lowe_filter_ratio)

        # Homography
        if len(filtered_matches) > args.min_matches:
            h, outliers = find_homography(keypoints, obj.keypoints, filtered_matches, args.robust_method, args.robust_threshold)
            h_inv = np.linalg.pinv(h)
            num_outliers = len(outliers)
            # Check minimum number of matches
            num_inliers = len(filtered_matches) - num_outliers
            if num_inliers < args.min_matches:
                h = None

        gaze = gaze_data.gaze_for_frame(frame_idx);
        if h is not None and len(gaze) > 0:
            width, height = resolution
            gaze_points = [ [ g['norm_pos_x'] * width, (1 - g['norm_pos_y']) * height ] for g in gaze ]
            gaze_points = np.float32(gaze_points).reshape(-1,1,2)
            gaze_points_undistorted = undistorter.undistort_points(gaze_points)
            object_points = cv2.perspectiveTransform(gaze_points_undistorted, h_inv)
            for i in range(0, len(gaze)):
                gaze[i]['object_x'] = object_points[i][0][0]
                gaze[i]['object_y'] = object_points[i][0][1]
            data_writer.write(gaze)
            if args.show_object:
                cv2.polylines(object_image, [np.int32(object_points)], False, (255, 0, 0), 3, cv2.LINE_AA)
            out_frame = cv2.polylines(out_frame, [np.int32(gaze_points_undistorted)], False, (255, 0, 0), 3, cv2.LINE_AA)

        # Draw video keypoints, matches, outliers
        if args.show_keypoints:
            filtered_points = [ keypoints[m.queryIdx] for m in filtered_matches ]
            out_frame = cv2.drawKeypoints(out_frame, keypoints, 0, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            out_frame = cv2.drawKeypoints(out_frame, filtered_points, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            if num_outliers > 0:
                outlier_points = [ keypoints[m.queryIdx] for m in outliers ]
                out_frame = cv2.drawKeypoints(out_frame, outlier_points, 0, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # Draw object keypoints, matches, outliers
        if args.show_object_keypoints:
            filtered_points = [ obj.keypoints[m.trainIdx] for m in filtered_matches ]
            object_image = cv2.drawKeypoints(object_image, obj.keypoints, 0, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            object_image = cv2.drawKeypoints(object_image, filtered_points, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            if num_outliers > 0:
                outlier_points = [ obj.keypoints[m.trainIdx] for m in outliers ]
                object_image = cv2.drawKeypoints(object_image, outlier_points, 0, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # Draw bounding box
        if h is not None:
            out_frame = draw_bounding_box(out_frame, h, obj.w, obj.h, (0, 0, 255), 2)

        video_text  = f"Frame {frame_idx}\n"
        video_text += f"Keypoints (green): {len(keypoints)}\n"
        video_text += f"Matches (red): {len(filtered_matches)}\n"
        video_text += f"Outliers (blue): {num_outliers}\n"
        video_text += f"Inliers: {num_inliers}\n"
        draw_text(out_frame, video_text, 16, 30)

        video_out.write(out_frame)
        if object_image is not None and args.show_object:
            cv2.imshow('Object', object_image)
        cv2.imshow('Frame', out_frame)
        print(f"Frame {frame_idx}, keypoints: {len(keypoints)}, matches: {len(filtered_matches)}     \r", end='')

        # Process keys
        should_stop = handle_events()
        if should_stop:
            break

    data_writer.close()
    video_out.release()


if __name__ == "__main__":
    main()
