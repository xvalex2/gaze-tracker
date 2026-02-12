import cv2
import numpy as np
import csv
import argparse
from os.path import isfile


def is_continuous(p1, p2):
    if p1['confidence'] == 0 or p2['confidence'] == 0:
        return False
    if p2['frame'] - p1['frame'] > 1:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        prog='trajectory',
        description='Draws gaze trajectory in the object plane.')
    parser.add_argument('object',    help='Image of tracked object')
    parser.add_argument('gaze',      help='Object plane gaze data CSV')
    parser.add_argument('out_image', help='Resulting image')

    args = parser.parse_args()

    source_files = (args.object, args.gaze)
    for f in source_files:
        if not isfile(f):
            print(f"Error: file does not extst: {f}")
            exit(-1)

    data, data_len = None, 0
    with open(args.gaze) as csvfile:
        csvfile.readline() # skip header
        fieldnames = ['timestamp', 'confidence', 'norm_pos_x', 'norm_pos_y', 'frame', 'frame_timestamp', 'object_x', 'object_y']
        reader = csv.DictReader(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        data = [row for row in reader]
        data_len = len(data) - 1

    segments = []
    i = 0
    while i < data_len:
        while i < data_len and not is_continuous(data[i], data[i+1]):
            i += 1
        if i >= data_len:
            break
        segment_start = i
        while i < data_len and is_continuous(data[i], data[i+1]):
            i += 1
        segments.append(data[segment_start:i])
        print(f"Segment from row {segment_start} ({data[segment_start]['timestamp']:.3f}s) to row {i} ({data[i]['timestamp']:.3f}s)")

    image = cv2.imread(args.object)
    h, w, _ = image.shape
    for segment in segments:
        points = [ [ p['object_x'], p['object_y'] ] for p in segment ]
        cv2.polylines(image, [np.int32(points)], False, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(args.out_image, image)


if __name__ == "__main__":
    main()
