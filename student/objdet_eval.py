import numpy as np
import matplotlib
matplotlib.use('wxagg')
import matplotlib.pyplot as plt
import torch
from shapely.geometry import Polygon
from operator import itemgetter
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.objdet_tools as tools


def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):
    """Compute per-frame detection performance metrics.
    
    For each valid ground-truth label, finds the best-matching detection by IoU.
    Uses shapely polygon intersection for accurate rotated bounding box overlap.
    
    Returns: [ious, center_devs, pos_negs]
    """
    true_positives = 0
    center_devs = []
    ious = []

    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []

        if valid:
            label_corners = tools.compute_box_corners(
                label.box.center_x, label.box.center_y,
                label.box.width, label.box.length, label.box.heading
            )

            for detection in detections:
                det_corners = tools.compute_box_corners(
                    detection.box.center_x, detection.box.center_y,
                    detection.box.width, detection.box.length, detection.box.heading
                )

                dist_x = abs(label.box.center_x - detection.box.center_x)
                dist_y = abs(label.box.center_y - detection.box.center_y)
                dist_z = abs(label.box.center_z - detection.box.center_z)

                poly_label = Polygon(label_corners)
                poly_det   = Polygon(det_corners)
                intersection = poly_label.intersection(poly_det).area
                union = poly_label.area + poly_det.area - intersection
                iou = intersection / union

                if iou >= min_iou:
                    matches_lab_det.append([iou, dist_x, dist_y, dist_z])
                    true_positives += 1

        if matches_lab_det:
            best_match = max(matches_lab_det, key=itemgetter(1))
            ious.append(best_match[0])
            center_devs.append(best_match[1:])

    # Precision/recall counts — in progress
    all_positives  = 0
    false_negatives = 0
    false_positives = 0

    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    return [ious, center_devs, pos_negs]


def compute_performance_stats(det_performance_all):
    """Aggregate per-frame metrics and plot detection performance histograms."""
    ious, center_devs, pos_negs = [], [], []
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])

    # Precision/recall — in progress
    precision = 0.0
    recall    = 0.0
    print(f'precision = {precision}, recall = {recall}')

    ious_all   = [e for tupl in ious for e in tupl]
    devs_x_all, devs_y_all, devs_z_all = [], [], []
    for tupl in center_devs:
        for elem in tupl:
            dx, dy, dz = elem
            devs_x_all.append(dx)
            devs_y_all.append(dy)
            devs_z_all.append(dz)

    data   = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = [
        'detection precision', 'detection recall', 'intersection over union',
        'position errors in X', 'position errors in Y', 'position errors in Z'
    ]
    textboxes = ['', '', '',
        '\n'.join([
            r'$\mathrm{mean}=%.4f$' % np.mean(devs_x_all),
            r'$\mathrm{sigma}=%.4f$' % np.std(devs_x_all),
            r'$\mathrm{n}=%.0f$' % len(devs_x_all)
        ]),
        '\n'.join([
            r'$\mathrm{mean}=%.4f$' % np.mean(devs_y_all),
            r'$\mathrm{sigma}=%.4f$' % np.std(devs_y_all),
            r'$\mathrm{n}=%.0f$' % len(devs_y_all)
        ]),
        '\n'.join([
            r'$\mathrm{mean}=%.4f$' % np.mean(devs_z_all),
            r'$\mathrm{sigma}=%.4f$' % np.std(devs_z_all),
            r'$\mathrm{n}=%.0f$' % len(devs_z_all)
        ])
    ]

    f, axes = plt.subplots(2, 3)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, ax in enumerate(axes.ravel()):
        ax.hist(data[idx], bins=20)
        ax.set_title(titles[idx])
        if textboxes[idx]:
            ax.text(0.05, 0.95, textboxes[idx], transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()