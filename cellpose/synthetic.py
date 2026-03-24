"""
Utilities for generating Cellpose-compatible synthetic segmentation datasets.

This module focuses on producing EM-like 2D grayscale images paired with exact
instance masks saved as ``*_masks.tif``. The generated folder structure matches
the existing Cellpose training loader, so fine-tuning can use the synthetic data
without changes to the training pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from scipy import ndimage as ndi
from tqdm import tqdm


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    out_dir: Path
    n_train: int
    n_val: int
    image_size: tuple[int, int] = (512, 512)
    seed: int = 0
    min_objects: int = 5
    max_objects: int = 12
    min_length: float = 40.0
    max_length: float = 150.0
    min_width: float = 8.0
    max_width: float = 22.0
    min_gap: int = 6
    near_touching_fraction: float = 0.25
    hard_case_fraction: float = 0.30
    blur_min: float = 0.6
    blur_max: float = 1.6
    noise_min: float = 0.02
    noise_max: float = 0.06
    boundary_contrast_min: float = 0.14
    boundary_contrast_max: float = 0.32
    show_progress: bool = True

    def __post_init__(self) -> None:
        if self.n_train < 0 or self.n_val < 0:
            raise ValueError("n_train and n_val must be non-negative")
        if self.image_size[0] < 64 or self.image_size[1] < 64:
            raise ValueError("image_size must be at least 64x64")
        if self.min_objects < 1 or self.max_objects < self.min_objects:
            raise ValueError("object count range is invalid")
        if self.min_length <= 0 or self.max_length < self.min_length:
            raise ValueError("length range is invalid")
        if self.min_width <= 0 or self.max_width < self.min_width:
            raise ValueError("width range is invalid")
        if self.min_gap < 0:
            raise ValueError("min_gap must be non-negative")
        _validate_fraction("near_touching_fraction", self.near_touching_fraction)
        _validate_fraction("hard_case_fraction", self.hard_case_fraction)
        if self.blur_min < 0 or self.blur_max < self.blur_min:
            raise ValueError("blur range is invalid")
        if self.noise_min < 0 or self.noise_max < self.noise_min:
            raise ValueError("noise range is invalid")
        if (self.boundary_contrast_min < 0 or
                self.boundary_contrast_max < self.boundary_contrast_min):
            raise ValueError("boundary contrast range is invalid")


def _validate_fraction(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def generate_dataset(config: SyntheticDatasetConfig) -> dict[str, list[dict[str, Any]]]:
    """Generate train/val splits and return per-split manifests."""
    config.out_dir.mkdir(parents=True, exist_ok=True)
    _write_config(config)

    manifests = {
        "train": _generate_split(config, "train", config.n_train, split_offset=0),
        "val": _generate_split(config, "val", config.n_val, split_offset=1_000_000),
    }
    return manifests


def _write_config(config: SyntheticDatasetConfig) -> None:
    payload = asdict(config)
    payload["out_dir"] = str(config.out_dir)
    payload["image_size"] = list(config.image_size)
    with open(config.out_dir / "synthetic_config.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _generate_split(config: SyntheticDatasetConfig, split: str, count: int,
                    split_offset: int) -> list[dict[str, Any]]:
    split_dir = config.out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    progress = tqdm(
        range(count),
        desc=f"{split} samples",
        unit="sample",
        disable=not config.show_progress,
        leave=True,
    )
    for index in progress:
        sample_seed = config.seed + split_offset + index
        rng = np.random.default_rng(sample_seed)
        sample_id = f"sample_{index:05d}"
        image, labels, metadata = generate_sample(rng, config, split=split)

        image_path = split_dir / f"{sample_id}.tif"
        mask_path = split_dir / f"{sample_id}_masks.tif"
        tifffile.imwrite(image_path, image)
        tifffile.imwrite(mask_path, labels)

        manifest.append({
            "id": sample_id,
            "split": split,
            "seed": sample_seed,
            "image_path": str(image_path.relative_to(config.out_dir)),
            "mask_path": str(mask_path.relative_to(config.out_dir)),
            **metadata,
        })
        progress.set_postfix(objects=metadata["num_objects"],
                             hard=int(metadata["hard_case"]))

    manifest_path = config.out_dir / f"{split}_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for item in manifest:
            fh.write(json.dumps(item, sort_keys=True))
            fh.write("\n")

    return manifest


def generate_sample(rng: np.random.Generator, config: SyntheticDatasetConfig,
                    split: str = "train") -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate one grayscale image and its exact instance mask."""
    hard_case_bias = 0.15 if split == "val" else 0.0
    hard_case = bool(rng.random() < min(1.0, config.hard_case_fraction + hard_case_bias))
    labels, object_summaries = _generate_labels(rng, config, hard_case=hard_case)
    image, render_meta = _render_em_like_image(rng, labels, config, hard_case=hard_case)
    metadata = {
        "hard_case": hard_case,
        "image_size": list(config.image_size),
        "num_objects": int(labels.max()),
        "objects": object_summaries,
        **render_meta,
    }
    return image, labels, metadata


def _generate_labels(rng: np.random.Generator, config: SyntheticDatasetConfig,
                     hard_case: bool) -> tuple[np.ndarray, list[dict[str, Any]]]:
    height, width = config.image_size
    labels = np.zeros((height, width), dtype=np.uint16)
    object_summaries: list[dict[str, Any]] = []

    target_count = int(rng.integers(config.min_objects, config.max_objects + 1))
    if hard_case:
        target_count = min(config.max_objects, target_count + int(rng.integers(0, 3)))

    placed = 0
    max_attempts = max(40, target_count * 30)
    for _ in range(max_attempts):
        if placed >= target_count:
            break

        near_touching = bool(hard_case and rng.random() < config.near_touching_fraction)
        gap = 1 if near_touching else config.min_gap
        candidate, summary = _generate_object_mask(
            rng,
            config.image_size,
            min_length=config.min_length,
            max_length=config.max_length,
            min_width=config.min_width,
            max_width=config.max_width,
            hard_case=hard_case,
        )
        if candidate is None:
            continue

        forbidden = labels > 0
        if gap > 0:
            forbidden = ndi.binary_dilation(forbidden, iterations=gap)
        if np.any(candidate & forbidden):
            continue

        area = int(candidate.sum())
        if area < 40:
            continue

        placed += 1
        labels[candidate] = placed
        summary["area"] = area
        summary["near_touching"] = near_touching
        object_summaries.append(summary)

    if placed == 0:
        fallback, summary = _generate_object_mask(
            rng,
            config.image_size,
            min_length=max(20.0, config.min_length * 0.5),
            max_length=max(30.0, config.min_length),
            min_width=max(4.0, config.min_width * 0.5),
            max_width=max(8.0, config.min_width),
            hard_case=False,
        )
        if fallback is not None:
            labels[fallback] = 1
            summary["area"] = int(fallback.sum())
            summary["near_touching"] = False
            object_summaries.append(summary)

    return labels, object_summaries


def _generate_object_mask(rng: np.random.Generator, image_size: tuple[int, int],
                          min_length: float, max_length: float, min_width: float,
                          max_width: float,
                          hard_case: bool) -> tuple[np.ndarray | None, dict[str, Any]]:
    height, width = image_size
    family = "tube" if rng.random() < 0.75 else "blob"

    base_width = float(rng.uniform(min_width, max_width))
    if family == "blob":
        length = float(rng.uniform(base_width * 2.0, max(base_width * 4.5, min_length)))
    else:
        length = float(rng.uniform(min_length, max_length))

    n_nodes = int(np.clip(round(length / 10.0), 6, 18))
    step = max(3.0, length / max(1, n_nodes - 1))
    heading = float(rng.uniform(0.0, 2.0 * np.pi))
    curvature_scale = float(rng.uniform(0.05, 0.20 if hard_case else 0.12))

    angle_noise = ndi.gaussian_filter1d(rng.normal(size=n_nodes - 1), sigma=1.0)
    angles = heading + np.cumsum(angle_noise * curvature_scale)
    points = np.zeros((n_nodes, 2), dtype=np.float32)
    for idx in range(1, n_nodes):
        delta = np.array([np.cos(angles[idx - 1]), np.sin(angles[idx - 1])],
                         dtype=np.float32)
        points[idx] = points[idx - 1] + step * delta
    points -= points.mean(axis=0, keepdims=True)

    profile = np.sin(np.linspace(0.0, np.pi, n_nodes, dtype=np.float64))
    profile = np.clip(profile, 0.0, None)**0.75
    profile = 0.35 + 0.65 * profile
    width_noise = ndi.gaussian_filter1d(rng.normal(scale=0.12, size=n_nodes), sigma=1.0)
    radii = base_width * np.clip(profile * (1.0 + width_noise), 0.25, 1.40)

    tangents = np.zeros_like(points)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    tangent_norm = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-6
    normals = np.stack((-tangents[:, 1], tangents[:, 0]), axis=1) / tangent_norm
    lateral_scale = 1.8 if hard_case else 0.8
    lateral_offsets = ndi.gaussian_filter1d(
        rng.normal(scale=lateral_scale, size=n_nodes), sigma=1.0)
    points = points + normals * lateral_offsets[:, None]

    margin = int(np.ceil(radii.max() + 4))
    min_xy = points.min(axis=0) - radii.max() - margin
    max_xy = points.max(axis=0) + radii.max() + margin
    span = max_xy - min_xy
    available = np.array([width - 2 * margin, height - 2 * margin], dtype=np.float32)
    if np.any(span <= 0):
        return None, {}

    if np.any(span > available):
        scale = float(0.95 * np.min(available / np.maximum(span, 1.0)))
        if scale < 0.35:
            return None, {}
        points *= scale
        radii *= scale
        min_xy = points.min(axis=0) - radii.max() - margin
        max_xy = points.max(axis=0) + radii.max() + margin

    x_shift_low = margin - min_xy[0]
    x_shift_high = (width - margin) - max_xy[0]
    y_shift_low = margin - min_xy[1]
    y_shift_high = (height - margin) - max_xy[1]
    if x_shift_low >= x_shift_high or y_shift_low >= y_shift_high:
        return None, {}

    shift = np.array([
        rng.uniform(x_shift_low, x_shift_high),
        rng.uniform(y_shift_low, y_shift_high),
    ], dtype=np.float32)
    points = points + shift

    mask = np.zeros((height, width), dtype=np.uint8)
    for idx, point in enumerate(points):
        center = tuple(np.round(point).astype(np.int32))
        radius = max(1, int(round(radii[idx])))
        cv2.circle(mask, center, radius, 1, thickness=-1)
        if idx > 0:
            prev_center = tuple(np.round(points[idx - 1]).astype(np.int32))
            thickness = max(1, int(round(min(radii[idx], radii[idx - 1]) * 2.0)))
            cv2.line(mask, prev_center, center, 1, thickness=thickness)

    mask = ndi.binary_fill_holes(mask > 0)
    if hard_case and rng.random() < 0.25:
        mask = ndi.binary_closing(mask, iterations=1)

    summary = {
        "family": family,
        "length": round(length, 2),
        "base_width": round(base_width, 2),
        "curvature_scale": round(curvature_scale, 4),
    }
    return mask.astype(bool), summary


def _render_em_like_image(rng: np.random.Generator, labels: np.ndarray,
                          config: SyntheticDatasetConfig,
                          hard_case: bool) -> tuple[np.ndarray, dict[str, Any]]:
    shape = labels.shape
    foreground = labels > 0
    y_grid, x_grid = np.mgrid[0:shape[0], 0:shape[1]]
    x_norm = (x_grid / max(1, shape[1] - 1) - 0.5).astype(np.float32)
    y_norm = (y_grid / max(1, shape[0] - 1) - 0.5).astype(np.float32)

    illumination_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    illumination = (np.cos(illumination_angle) * x_norm +
                    np.sin(illumination_angle) * y_norm)

    background = (
        0.58
        + 0.07 * _scaled_field(rng, shape, sigma=float(rng.uniform(18.0, 40.0)))
        + 0.03 * _scaled_field(rng, shape, sigma=float(rng.uniform(5.0, 10.0)))
        + 0.08 * illumination
    )
    image = background.astype(np.float32)

    boundary_map = np.zeros(shape, dtype=np.float32)
    object_count = int(labels.max())
    for label_id in range(1, object_count + 1):
        mask = labels == label_id
        dist_inside = ndi.distance_transform_edt(mask)
        coarse_texture = _scaled_field(rng, shape, sigma=float(rng.uniform(4.0, 10.0)))
        fine_texture = _scaled_field(rng, shape, sigma=float(rng.uniform(1.0, 2.5)))
        base_intensity = float(rng.uniform(0.38, 0.62))
        image[mask] = (
            base_intensity
            + 0.06 * coarse_texture[mask]
            + 0.025 * fine_texture[mask]
        )

        boundary_width = float(rng.uniform(0.8, 2.4 if hard_case else 1.8))
        boundary_map += np.exp(-((dist_inside - 1.0)**2) /
                               (2.0 * boundary_width * boundary_width)).astype(np.float32) * mask
        _add_internal_inclusions(rng, image, mask, dist_inside)

    if object_count > 0:
        boundary_map = np.clip(boundary_map, 0.0, 1.0)

    edge_shadow = ndi.gaussian_filter(foreground.astype(np.float32),
                                      sigma=float(rng.uniform(1.2, 2.8)))
    boundary_contrast = float(
        rng.uniform(config.boundary_contrast_min, config.boundary_contrast_max))
    image -= boundary_contrast * boundary_map
    image -= 0.05 * edge_shadow

    blur_sigma_y = float(rng.uniform(config.blur_min, config.blur_max))
    blur_sigma_x = float(rng.uniform(config.blur_min, config.blur_max))
    image = ndi.gaussian_filter(image, sigma=(blur_sigma_y, blur_sigma_x))

    poisson_scale = float(rng.uniform(180.0, 420.0))
    image = np.clip(image, 0.0, 1.0)
    image = rng.poisson(image * poisson_scale).astype(np.float32) / poisson_scale

    noise_sigma = float(rng.uniform(config.noise_min, config.noise_max))
    image += noise_sigma * rng.normal(size=shape).astype(np.float32)
    image += (noise_sigma * 0.5) * _scaled_field(rng, shape, sigma=0.75)

    image = _normalize_to_uint16(image)
    render_meta = {
        "blur_sigma": [round(blur_sigma_y, 3), round(blur_sigma_x, 3)],
        "noise_sigma": round(noise_sigma, 4),
        "boundary_contrast": round(boundary_contrast, 4),
        "poisson_scale": round(poisson_scale, 1),
    }
    return image, render_meta


def _add_internal_inclusions(rng: np.random.Generator, image: np.ndarray, mask: np.ndarray,
                             dist_inside: np.ndarray) -> None:
    max_radius = min(8.0, max(2.5, float(dist_inside.max()) / 3.0))
    if max_radius < 2.5:
        return

    num_spots = int(rng.integers(0, 4))
    if num_spots == 0:
        return

    valid_centers = np.argwhere(dist_inside > 3.0)
    if len(valid_centers) == 0:
        return

    for _ in range(num_spots):
        center_y, center_x = valid_centers[int(rng.integers(0, len(valid_centers)))]
        radius = max(2, int(round(rng.uniform(2.0, max_radius))))
        inclusion = np.zeros(mask.shape, dtype=np.uint8)
        cv2.circle(inclusion, (int(center_x), int(center_y)), radius, 1, thickness=-1)
        inclusion_mask = inclusion.astype(bool) & mask
        if not np.any(inclusion_mask):
            continue
        image[inclusion_mask] -= float(rng.uniform(0.08, 0.18))


def _scaled_field(rng: np.random.Generator, shape: tuple[int, int],
                  sigma: float) -> np.ndarray:
    field = ndi.gaussian_filter(rng.normal(size=shape).astype(np.float32), sigma=sigma)
    field -= field.mean()
    std = float(field.std())
    if std < 1e-6:
        return np.zeros(shape, dtype=np.float32)
    return field / std


def _normalize_to_uint16(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [1.0, 99.0])
    if hi <= lo:
        hi = lo + 1e-6
    image = np.clip(image, lo, hi)
    image = (image - lo) / (hi - lo)
    return np.round(image * 65535.0).astype(np.uint16)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate EM-like synthetic image/mask pairs for Cellpose training.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory for train/val splits.")
    parser.add_argument("--n-train", default=5000, type=int,
                        help="Number of synthetic training samples to generate.")
    parser.add_argument("--n-val", default=500, type=int,
                        help="Number of synthetic validation samples to generate.")
    parser.add_argument("--image-size", default=[512, 512], type=int, nargs="+",
                        help="Image size as one integer or two integers: H [W].")
    parser.add_argument("--seed", default=0, type=int,
                        help="Base random seed.")
    parser.add_argument("--min-objects", default=5, type=int,
                        help="Minimum number of instances per image.")
    parser.add_argument("--max-objects", default=12, type=int,
                        help="Maximum number of instances per image.")
    parser.add_argument("--min-length", default=40.0, type=float,
                        help="Minimum centerline length for elongated objects.")
    parser.add_argument("--max-length", default=150.0, type=float,
                        help="Maximum centerline length for elongated objects.")
    parser.add_argument("--min-width", default=8.0, type=float,
                        help="Minimum object half-width in pixels.")
    parser.add_argument("--max-width", default=22.0, type=float,
                        help="Maximum object half-width in pixels.")
    parser.add_argument("--min-gap", default=6, type=int,
                        help="Minimum pixel gap between most objects.")
    parser.add_argument("--near-touching-fraction", default=0.25, type=float,
                        help="Fraction of hard-case objects allowed to be near-touching.")
    parser.add_argument("--hard-case-fraction", default=0.30, type=float,
                        help="Fraction of samples biased toward harder geometry/rendering.")
    parser.add_argument("--blur-min", default=0.6, type=float,
                        help="Minimum Gaussian blur sigma.")
    parser.add_argument("--blur-max", default=1.6, type=float,
                        help="Maximum Gaussian blur sigma.")
    parser.add_argument("--noise-min", default=0.02, type=float,
                        help="Minimum additive noise scale.")
    parser.add_argument("--noise-max", default=0.06, type=float,
                        help="Maximum additive noise scale.")
    parser.add_argument("--boundary-contrast-min", default=0.14, type=float,
                        help="Minimum darkening applied along instance boundaries.")
    parser.add_argument("--boundary-contrast-max", default=0.32, type=float,
                        help="Maximum darkening applied along instance boundaries.")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable tqdm progress bars during synthetic generation.")
    return parser


def _parse_image_size(values: list[int]) -> tuple[int, int]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2:
        return values[0], values[1]
    raise ValueError("image-size expects one integer or two integers")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = SyntheticDatasetConfig(
        out_dir=args.out,
        n_train=args.n_train,
        n_val=args.n_val,
        image_size=_parse_image_size(args.image_size),
        seed=args.seed,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        min_length=args.min_length,
        max_length=args.max_length,
        min_width=args.min_width,
        max_width=args.max_width,
        min_gap=args.min_gap,
        near_touching_fraction=args.near_touching_fraction,
        hard_case_fraction=args.hard_case_fraction,
        blur_min=args.blur_min,
        blur_max=args.blur_max,
        noise_min=args.noise_min,
        noise_max=args.noise_max,
        boundary_contrast_min=args.boundary_contrast_min,
        boundary_contrast_max=args.boundary_contrast_max,
        show_progress=not args.no_progress,
    )
    generate_dataset(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
