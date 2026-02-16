import numpy as np
import cv2
import csv
import argparse
from os.path import isfile


DEFAULT_SIGMA = 10
DEFAULT_ALPHA = 0.5


def main():
    parser = argparse.ArgumentParser(
        prog='heatmap',
        description='Draws a dwell-time heatmap in the object plane.')
    parser.add_argument('object',    help='Image of the tracked object.')
    parser.add_argument('gaze',      help='Gaze data in the object plane (CSV file).')
    parser.add_argument('out_image', help='Output image.')
    parser.add_argument('--alpha', required=False, default=DEFAULT_ALPHA, type=float,
        help=f'Heatmap transparency over the original image. Default is {DEFAULT_ALPHA}.')
    parser.add_argument('--sigma', required=False, default=DEFAULT_SIGMA, type=float,
        help=f'Gaussian blur sigma for the heatmap. Default is {DEFAULT_SIGMA}.')
    parser.add_argument('--disable-confidence-filter', action='store_true', 
        help='Do not filter out gaze points with zero confidence.')

    args = parser.parse_args()

    source_files = (args.object, args.gaze)
    for f in source_files:
        if not isfile(f):
            print(f"Error: file does not extst: {f}")
            exit(-1)

    data = None
    with open(args.gaze) as csvfile:
        csvfile.readline() # skip header
        fieldnames = [ 'timestamp', 'confidence', 'norm_pos_x', 'norm_pos_y', 
                       'frame', 'frame_timestamp', 
                       'object_x', 'object_y', 'object_norm_x', 'object_norm_y']
        reader = csv.DictReader(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        data = [row for row in reader]
        data_len = len(data)

    image = cv2.imread(args.object)
    h, w, _ = image.shape

    x = [ row['object_norm_x'] * w for row in data if row['confidence'] > 0 or args.disable_confidence_filter ]
    y = [ (1.0 - row['object_norm_y']) * h for row in data if row['confidence'] > 0 or args.disable_confidence_filter ]

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=(w, h), range=[[0, w], [0, h]])
    extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
    heatmap = heatmap.T.astype(np.float32) # float32 for OpenCV compatibility

    ksize = int(6*args.sigma + 1)
    if ksize % 2 == 0:
       ksize += 1
    heatmap = cv2.GaussianBlur(heatmap, (ksize, ksize), sigmaX=args.sigma, sigmaY=args.sigma)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    out_image = cv2.addWeighted(heatmap, args.alpha, image, 1-args.alpha, 0)
    cv2.imwrite(args.out_image, out_image)


if __name__ == "__main__":
    main()
