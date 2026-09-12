"""End-to-end smoke test for the full detection -> segmentation ->
validation -> tracing -> formatting pipeline.

Runs against whatever real case data is present under app/data/ (that
directory is gitignored -- it holds real CT volumes -- so this test
skips itself when none is available, e.g. in CI, rather than failing).
"""
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

DATA_DIR = APP_DIR / "data"
CASES = sorted(DATA_DIR.glob("*/orig*.nii")) if DATA_DIR.exists() else []


def _mask_for(ct_path: Path) -> Path:
    suffix = ct_path.stem.replace("orig", "")
    return ct_path.with_name(f"mask{suffix}.nii")


@pytest.mark.skipif(not CASES, reason="no sample case data under backend/app/data/")
@pytest.mark.parametrize("ct_path", CASES, ids=lambda p: p.parent.name)
def test_run_case_produces_valid_schema(ct_path):
    from pipeline.branch_detection import run_case

    mask_path = _mask_for(ct_path)
    assert mask_path.exists(), f"missing mask for {ct_path}"

    result = run_case(ct_path, mask_path)

    assert result["case_id"] == ct_path.parent.name
    assert result["parent"] == {"instance_id": "aorta"}
    assert isinstance(result["daughters"], list)

    seen_ids = set()
    for i, d in enumerate(result["daughters"], start=1):
        assert d["instance_id"] == f"branch_{i:03d}"
        assert d["instance_id"] not in seen_ids
        seen_ids.add(d["instance_id"])
        assert d["parent_instance_id"] == "aorta"

        for key in ("ostium_xyz_mm", "seed_xyz_mm", "direction_xyz"):
            assert isinstance(d[key], list) and len(d[key]) == 3
            assert all(isinstance(v, float) for v in d[key])

        assert isinstance(d["radius_mm"], float) and d["radius_mm"] > 0

        # direction_xyz must be a unit vector...
        dir_len = sum(v * v for v in d["direction_xyz"]) ** 0.5
        assert abs(dir_len - 1.0) < 1e-6

        # ...and the seed must sit exactly 5 mm from the ostium along it,
        # per the challenge's daughter-seed definition.
        offset = [s - o for s, o in zip(d["seed_xyz_mm"], d["ostium_xyz_mm"])]
        offset_len = sum(v * v for v in offset) ** 0.5
        assert abs(offset_len - 5.0) < 1e-3
