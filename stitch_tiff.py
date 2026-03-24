import argparse
from pathlib import Path

import numpy as np
import tifffile

try:
    from scipy.ndimage import binary_fill_holes
except ImportError:
    binary_fill_holes = None


def imread_tiff(path):
    with tifffile.TiffFile(path) as tif:
        return tif.asarray()


def imsave_tiff(path, array):
    tifffile.imwrite(path, data=array, compression="zlib")


def _intersection_over_union(masks_a, masks_b):
    masks_a = np.asarray(masks_a, dtype=np.int64)
    masks_b = np.asarray(masks_b, dtype=np.int64)
    n_a = int(masks_a.max()) + 1
    n_b = int(masks_b.max()) + 1
    overlap = np.bincount(
        masks_a.ravel() * n_b + masks_b.ravel(),
        minlength=n_a * n_b,
    ).reshape(n_a, n_b)
    area_a = overlap.sum(axis=1, keepdims=True)
    area_b = overlap.sum(axis=0, keepdims=True)
    union = area_a + area_b - overlap
    iou = np.zeros_like(overlap, dtype=np.float64)
    valid = union > 0
    iou[valid] = overlap[valid] / union[valid]
    return iou


def _assign_new_labels(frame, start_label, dtype):
    count = int(frame.max())
    if count == 0:
        return frame, start_label
    relabel = np.arange(start_label + 1, start_label + count + 1, dtype=dtype)
    relabel = np.append(np.array(0, dtype=dtype), relabel)
    return relabel[frame], start_label + count


def _match_frame_to_previous(current_frame, previous_frame, stitch_threshold, next_label):
    iou = _intersection_over_union(current_frame, previous_frame)[1:, 1:]
    if not iou.size:
        return None, next_label

    n_current = iou.shape[0]
    mapping = np.zeros(n_current, dtype=current_frame.dtype)
    used_previous = set()

    # Match strongest overlaps first and enforce one-to-one label reuse.
    candidates = np.argwhere(iou >= stitch_threshold)
    if candidates.size:
        scores = iou[candidates[:, 0], candidates[:, 1]]
        order = np.argsort(scores)[::-1]
        for idx in order:
            cur_idx, prev_idx = candidates[idx]
            if mapping[cur_idx] != 0 or prev_idx in used_previous:
                continue
            mapping[cur_idx] = prev_idx + 1
            used_previous.add(prev_idx)

    missing = np.nonzero(mapping == 0)[0]
    if missing.size:
        mapping[missing] = np.arange(
            next_label + 1,
            next_label + len(missing) + 1,
            dtype=current_frame.dtype,
        )
        next_label += len(missing)

    mapping = np.append(np.array(0, dtype=current_frame.dtype), mapping)
    return mapping[current_frame], next_label


def stitch3d(masks, stitch_threshold=0.25, max_frame_gap=0):
    masks = np.asarray(masks)
    if masks.ndim != 3:
        raise ValueError(f"Expected a 3D mask stack; got shape {masks.shape}")
    if max_frame_gap < 0:
        raise ValueError("max_frame_gap must be >= 0")

    stitched = masks.copy()
    stitched[0], next_label = _assign_new_labels(stitched[0], 0, stitched.dtype)
    for i in range(1, len(stitched)):
        if stitched[i].max() == 0:
            continue

        max_lookback = min(i, max_frame_gap + 1)
        relabeled = None
        candidate_next_label = next_label
        for gap in range(1, max_lookback + 1):
            previous = stitched[i - gap]
            if previous.max() == 0:
                continue
            relabeled, candidate_next_label = _match_frame_to_previous(
                stitched[i], previous, stitch_threshold, next_label
            )
            if relabeled is not None:
                break

        if relabeled is None:
            stitched[i], next_label = _assign_new_labels(stitched[i], next_label, stitched.dtype)
        else:
            stitched[i] = relabeled
            next_label = candidate_next_label
    return stitched


def _renumber_labels(masks):
    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]
    remapped = np.zeros_like(masks)
    for new_label, old_label in enumerate(unique_labels, start=1):
        remapped[masks == old_label] = new_label
    return remapped


def fill_holes_and_remove_small_masks(masks, min_size=0):
    masks = np.asarray(masks)
    output = np.zeros_like(masks)
    next_label = 1
    for label in np.unique(masks):
        if label == 0:
            continue
        region = masks == label
        if region.sum() < min_size:
            continue
        if binary_fill_holes is not None:
            region = binary_fill_holes(region)
        output[region] = next_label
        next_label += 1
    return output


def stitch_tiff(
    input_path,
    output_path=None,
    stitch_threshold=0.25,
    check_labels=True,
    max_frame_gap=0,
):
    """Load a 3D TIFF mask stack, stitch labels across planes, and optionally save it."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    masks = np.asarray(imread_tiff(str(input_path)))
    if masks.ndim != 3:
        raise ValueError(
            f"Expected a 3D TIFF mask stack with shape (z, y, x); got shape {masks.shape}"
        )

    if check_labels:
        if np.any(masks < 0):
            raise ValueError("Mask TIFF must contain non-negative labels.")
        if not np.issubdtype(masks.dtype, np.integer):
            if np.allclose(masks, np.round(masks)):
                masks = np.round(masks).astype(np.int32)
            else:
                raise ValueError("Mask TIFF must contain integer labels.")

    stitched = stitch3d(
        masks,
        stitch_threshold=stitch_threshold,
        max_frame_gap=max_frame_gap,
    )
    stitched = fill_holes_and_remove_small_masks(stitched, min_size=0)
    stitched = _renumber_labels(stitched)

    if output_path is None:
        suffix = input_path.suffix
        stem = input_path.name[:-len(suffix)] if suffix else input_path.name
        output_path = input_path.with_name(f"{stem}_stitched{suffix or '.tif'}")
    else:
        output_path = Path(output_path)

    imsave_tiff(str(output_path), stitched)
    return stitched, output_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Apply Cellpose 3D mask stitching directly to a TIFF label stack."
    )
    parser.add_argument("input_tif", help="Path to the input TIFF mask stack.")
    parser.add_argument(
        "-o",
        "--output",
        dest="output_tif",
        default=None,
        help="Path to save the stitched TIFF. Defaults to <input>_stitched.tif",
    )
    parser.add_argument(
        "--stitch-threshold",
        type=float,
        default=0.25,
        help="IoU threshold used to stitch labels across adjacent planes.",
    )
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=0,
        help="Allow linking across up to this many empty or missing intermediate frames.",
    )
    parser.add_argument(
        "--no-check-labels",
        action="store_true",
        help="Skip validation that the TIFF contains non-negative integer labels.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    _, output_path = stitch_tiff(
        args.input_tif,
        output_path=args.output_tif,
        stitch_threshold=args.stitch_threshold,
        check_labels=not args.no_check_labels,
        max_frame_gap=args.max_frame_gap,
    )
    print(f"Saved stitched mask stack to {output_path}")


if __name__ == "__main__":
    main()
