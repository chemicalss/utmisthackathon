"""Final formatting step: turns traced/measured branch dicts into the
exact submission schema from the challenge brief (case_id, parent, and a
daughters list with ostium/seed/radius/direction per instance)."""
from dataclasses import asdict

from models.schemas import BranchInstance


def _to_list(v) -> list[float]:
    return [float(x) for x in v]


def format_case_output(case_id: str, branches: list[dict]) -> dict:
    daughters = []
    for i, b in enumerate(branches, start=1):
        instance = BranchInstance(
            instance_id=f"branch_{i:03d}",
            parent_instance_id="aorta",
            ostium_xyz_mm=_to_list(b["ostium_xyz_mm"]),
            seed_xyz_mm=_to_list(b["seed_xyz_mm"]),
            radius_mm=float(b["radius_mm"]),
            direction_xyz=_to_list(b["direction_xyz"]),
        )
        daughters.append(asdict(instance))
    return {
        "case_id": case_id,
        "parent": {"instance_id": "aorta"},
        "daughters": daughters,
    }
