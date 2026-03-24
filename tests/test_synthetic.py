import json
import shutil
from pathlib import Path

from cellpose import io
from cellpose.synthetic import SyntheticDatasetConfig, generate_dataset


def test_generate_synthetic_dataset():
    root = Path("synthetic_test_workspace")
    shutil.rmtree(root, ignore_errors=True)
    out_dir = root / "synthetic"
    try:
        config = SyntheticDatasetConfig(
            out_dir=out_dir,
            n_train=2,
            n_val=1,
            image_size=(128, 128),
            seed=7,
            min_objects=3,
            max_objects=4,
            min_length=24.0,
            max_length=56.0,
            min_width=5.0,
            max_width=12.0,
            min_gap=3,
            show_progress=False,
        )

        manifests = generate_dataset(config)

        train_dir = out_dir / "train"
        val_dir = out_dir / "val"

        assert (out_dir / "synthetic_config.json").exists()
        assert (out_dir / "train_manifest.jsonl").exists()
        assert (out_dir / "val_manifest.jsonl").exists()
        assert len(list(train_dir.glob("*.tif"))) == 4
        assert len(list(val_dir.glob("*.tif"))) == 2
        assert len(manifests["train"]) == 2
        assert len(manifests["val"]) == 1

        sample_image = io.imread(train_dir / "sample_00000.tif")
        sample_mask = io.imread(train_dir / "sample_00000_masks.tif")
        assert sample_image.shape == (128, 128)
        assert sample_mask.shape == (128, 128)
        assert sample_mask.dtype.kind in {"u", "i"}
        assert sample_mask.max() >= 1

        output = io.load_train_test_data(str(train_dir), str(val_dir), mask_filter="_masks")
        train_images, train_labels, image_names, test_images, test_labels, test_image_names = output
        assert len(train_images) == 2
        assert len(train_labels) == 2
        assert len(image_names) == 2
        assert len(test_images) == 1
        assert len(test_labels) == 1
        assert len(test_image_names) == 1

        with open(out_dir / "train_manifest.jsonl", "r", encoding="utf-8") as fh:
            first_row = json.loads(fh.readline())
        assert first_row["id"] == "sample_00000"
        assert first_row["num_objects"] >= 1
    finally:
        shutil.rmtree(root, ignore_errors=True)
