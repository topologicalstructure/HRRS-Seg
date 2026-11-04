# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import random

import matplotlib.pyplot as plt
import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process Potsdam validation labels')
    parser.add_argument(
        '--src_dir',
        default='D:\\git\\HRRS-Seg\\data\\potsdam\\ann_dir\\val',
        help='directory that stores the original validation labels')
    parser.add_argument(
        '--dst_dir',
        default='D:\\git\\HRRS-Seg\\data\\potsdam\\ann_dir_processed\\val',
        help='directory that stores the processed validation labels')
    parser.add_argument(
        '--clutter_idx',
        type=int,
        default=0,
        help='label index that represents clutter/background')
    parser.add_argument(
        '--sample_count',
        type=int,
        default=5,
        help='number of random labels to visualize')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='random seed used to choose visualization samples')
    parser.add_argument(
        '--vis_path',
        default=None,
        help='path to save the before/after comparison figure')
    return parser.parse_args()


def _load_label(label_path):
    label = mmcv.imread(label_path, flag='unchanged')
    if label.ndim == 3:
        label = label[..., 0]
    return label.astype(np.uint8, copy=False)


def _replace_clutter_with_nearest(label, clutter_idx):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            'process_potsdam requires opencv-python-headless to be installed') from exc

    processed = label.copy()
    clutter_mask = processed == clutter_idx
    if not np.any(clutter_mask):
        print('No clutter pixels found in label.')
        return processed

    non_clutter_mask = ~clutter_mask
    if not np.any(non_clutter_mask):
        raise ValueError('The label contains only clutter pixels.')

    # OpenCV assigns a unique label to each zero pixel in DIST_LABEL_PIXEL mode.
    # We mark non-clutter pixels as zero so every clutter pixel can recover the
    # nearest non-clutter pixel and inherit its class id.
    src = np.where(clutter_mask, 255, 0).astype(np.uint8)
    _, nearest_labels = cv2.distanceTransformWithLabels(
        src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)

    reference_labels = nearest_labels[non_clutter_mask]
    reference_values = processed[non_clutter_mask]
    label_lookup = np.zeros(int(nearest_labels.max()) + 1, dtype=processed.dtype)
    label_lookup[reference_labels] = reference_values
    processed[clutter_mask] = label_lookup[nearest_labels[clutter_mask]]
    return processed


def _save_visualization(samples, vis_path):
    if not samples:
        return

    num_rows = len(samples)
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 4 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (filename, before, after) in enumerate(samples):
        ax_before, ax_after = axes[row]
        ax_before.imshow(before, cmap='tab10', vmin=0, vmax=5)
        ax_before.set_title(f'Before: {filename}')
        ax_before.axis('off')

        ax_after.imshow(after, cmap='tab10', vmin=0, vmax=5)
        ax_after.set_title(f'After: {filename}')
        ax_after.axis('off')

    plt.tight_layout()
    plt.savefig(vis_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def process_potsdam(src_dir=None,
                    dst_dir=None,
                    clutter_idx=5,
                    sample_count=5,
                    seed=42,
                    vis_path=None):
    src_dir = src_dir or osp.join('data', 'potsdam', 'ann_dir', 'val')
    dst_dir = dst_dir or osp.join('data', 'potsdam', 'ann_dir_processed', 'val')
    os.makedirs(dst_dir, exist_ok=True)

    label_paths = sorted(
        p for p in glob.glob(osp.join(src_dir, '*')) if osp.isfile(p))
    if not label_paths:
        raise FileNotFoundError(f'No labels found in {src_dir!r}.')

    rng = random.Random(seed)
    sample_paths = set(rng.sample(label_paths, min(sample_count, len(label_paths))))
    samples = []

    for label_path in label_paths:
        before = _load_label(label_path)
        after = _replace_clutter_with_nearest(before, clutter_idx)

        dst_path = osp.join(dst_dir, osp.basename(label_path))
        mmcv.imwrite(after.astype(np.uint8), dst_path)

        if label_path in sample_paths:
            samples.append((osp.basename(label_path), before, after))

    vis_path = vis_path or osp.join(dst_dir, 'potsdam_clutter_replacement.png')
    _save_visualization(samples, vis_path)
    return dst_dir, vis_path


def main():
    args = parse_args()
    dst_dir, vis_path = process_potsdam(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        clutter_idx=args.clutter_idx,
        sample_count=args.sample_count,
        seed=args.seed,
        vis_path=args.vis_path)
    print(f'Processed labels saved to {dst_dir}')
    print(f'Comparison figure saved to {vis_path}')


if __name__ == '__main__':
    main()
