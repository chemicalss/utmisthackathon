from detection.launch_points import find_launch_points
from detection.outward_search import walk_outward
from segmentation.geodesic_watershed import merge_nearby_launch_points, geodesic_watershed, export_branches
from validation.validator import validate_branches


def detect_and_segment(ct, aorta_mask):
    """Everything up to (not including) Proximal Tracing. Returns the
    'branches' list in the shared handoff format."""
    raw_launch_points = find_launch_points(ct, aorta_mask)
    launch_points = merge_nearby_launch_points(raw_launch_points)
    tracks = [walk_outward(ct, lp) for lp in launch_points]
    regions = geodesic_watershed(ct, aorta_mask, tracks)
    validated = validate_branches(regions)
    return export_branches(validated, ct)