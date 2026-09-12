from pathlib import Path

import numpy as np

from preprocessing.volume import load_volume
from detection.launch_points import find_launch_points
from detection.outward_search import walk_outward
from segmentation.geodesic_watershed import merge_nearby_launch_points, geodesic_watershed, export_branches
from validation.validator import validate_branches
from tracing.proximal import trace_proximal, centerline_to_physical
from tracing.stopping import stop_at_bifurcation
from output.formatter import format_case_output
from utils.geometry import point_at_arclength, unit_vector, local_radius_mm

# "Daughter seed" per the challenge brief: the centre of the daughter
# lumen, 5 mm outward from the ostium along the daughter path.
DAUGHTER_SEED_OFFSET_MM = 5.0


def detect_and_segment(ct, aorta_mask):
    """Everything up to (not including) proximal tracing. Returns the
    'branches' list in the shared handoff format: one dict per
    validated branch with branch_id, ostium_xyz_mm, voxels and
    geodesic_cost. seed_xyz_mm is filled in later, by trace_and_measure."""
    print("[1] Finding launch points on aortic surface...", flush=True)
    raw_launch_points = find_launch_points(ct, aorta_mask)
    print(f"[2] Found {len(raw_launch_points)} raw launch points, merging nearby...", flush=True)
    launch_points = merge_nearby_launch_points(raw_launch_points)
    print(f"[3] Merged to {len(launch_points)} unique launch points, walking outward...", flush=True)
    tracks = [walk_outward(ct, lp) for lp in launch_points]
    accepted = [t for t in tracks if t.accepted]
    print(f"[4] {len(accepted)} tracks accepted, running geodesic watershed...", flush=True)
    regions = geodesic_watershed(ct, aorta_mask, tracks)
    print(f"[5] Watershed created {len(regions)} regions, validating...", flush=True)
    validated = validate_branches(regions)
    print(f"[6] Validation passed {len(validated)} branches", flush=True)
    return export_branches(validated, ct)


def trace_and_measure(branch: dict, ct) -> dict:
    """Proximal tracing + bifurcation stop, then derive the three
    measurements the submission format needs: the daughter seed point,
    its local radius, and the initial direction out of the aorta."""
    traced = trace_proximal(branch, ct)
    stopped = stop_at_bifurcation(traced, ct)

    path_mm = centerline_to_physical(stopped["final_centerline"], ct)
    ostium_mm = np.array(branch["ostium_xyz_mm"], dtype=float)

    if path_mm:
        # Anchor arc length at the true (continuous) ostium centre rather
        # than the nearest whole branch voxel the centreline starts from.
        path_mm = [np.array(p, dtype=float) for p in path_mm]
        path_mm[0] = ostium_mm
    else:
        path_mm = [ostium_mm]

    seed_mm, _ = point_at_arclength(path_mm, DAUGHTER_SEED_OFFSET_MM)
    direction = unit_vector(seed_mm - ostium_mm)

    seed_voxel_xyz = ct.TransformPhysicalPointToIndex(tuple(seed_mm))
    radius_mm = local_radius_mm(branch["voxels"], seed_voxel_xyz, ct.GetSpacing())

    return {
        **stopped,
        "seed_xyz_mm": seed_mm,
        "direction_xyz": direction,
        "radius_mm": radius_mm,
    }


def run_case_from_volumes(ct, aorta_mask, case_id: str) -> dict:
    """Same pipeline as run_case, for callers (like run.py's visual-check
    step) that already have the CT and mask loaded and want to reuse
    them instead of reading the files twice."""
    print(f"\n=== Pipeline for {case_id} ===", flush=True)
    branches = detect_and_segment(ct, aorta_mask)
    print(f"\n[7] Starting proximal tracing on {len(branches)} branches...", flush=True)
    measured = [trace_and_measure(b, ct) for b in branches]
    print(f"[8] Tracing complete, formatting output...", flush=True)
    result = format_case_output(case_id, measured)
    print(f"[9] Done! Detected {len(result['daughters'])} daughter branches", flush=True)
    return result


def run_case(ct_path, mask_path, case_id: str | None = None) -> dict:
    """Full per-case pipeline: load -> detect & segment -> validate ->
    proximal trace -> measure -> format. This is the single entry point
    run.py calls for each subject, and what every backend stage above
    feeds into."""
    ct, aorta_mask = load_volume(ct_path, mask_path)
    if case_id is None:
        case_id = Path(ct_path).parent.name or Path(ct_path).stem
    return run_case_from_volumes(ct, aorta_mask, case_id)
