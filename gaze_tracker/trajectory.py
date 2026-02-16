import cv2
import numpy as np
import csv
import argparse
from os.path import isfile


def is_continuous(p1, p2, confidence_filter):
    if confidence_filter:
        if p1['confidence'] == 0 or p2['confidence'] == 0:
            return False
    if p2['frame'] - p1['frame'] > 1:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        prog='trajectory',
        description='Draws the gaze trajectory in the object plane.')
    parser.add_argument('object',    help='Image of the tracked object.')
    parser.add_argument('gaze',      help='Gaze data in the object plane (CSV file)')
    parser.add_argument('out_image', help='Output image.')
    parser.add_argument('--disable-confidence-filter', action='store_true', 
        help='Do not filter out gaze points with zero confidence.')

    args = parser.parse_args()

    source_files = (args.object, args.gaze)
    for f in source_files:
        if not isfile(f):
            print(f"Error: file does not extst: {f}")
            exit(-1)

    data, data_len = None, 0
    with open(args.gaze) as csvfile:
        csvfile.readline() # skip header
        fieldnames = [ 'timestamp', 'confidence', 'norm_pos_x', 'norm_pos_y', 
                       'frame', 'frame_timestamp', 
                       'object_x', 'object_y', 'object_norm_x', 'object_norm_y']
        reader = csv.DictReader(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        data = [row for row in reader]
        data_len = len(data) - 1

    i = 0
    segments = []
    confidence_filter = not args.disable_confidence_filter
    while i < data_len:
        while i < data_len and not is_continuous(data[i], data[i+1], confidence_filter):
            i += 1
        if i >= data_len:
            break
        segment_start = i
        while i < data_len and is_continuous(data[i], data[i+1], confidence_filter):
            i += 1
        segments.append(data[segment_start:i])
        print(f"Segment from row {segment_start} ({data[segment_start]['timestamp']:.3f}s) to row {i} ({data[i]['timestamp']:.3f}s)")

    image = cv2.imread(args.object)
    h, w, _ = image.shape
    for segment in segments:
        points = [ [ p['object_norm_x'] * w, (1.0 - p['object_norm_y']) * h ] for p in segment ]
        cv2.polylines(image, [np.int32(points)], False, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(args.out_image, image)


if __name__ == "__main__":
    main()
