from models.schemas import BranchRegion
from validation.persistence import check_persistence
from validation.direction import check_direction
from validation.continuity import check_continuity


def validate_branches(regions: list[BranchRegion],
                       min_persisted_mm: float = 5.0,
                       min_vesselness: float = 0.05) -> list[BranchRegion]:
    # min_vesselness is calibrated against compute_vesselness's own
    # percentile-normalised scale (see validation/vesselness.py), where
    # 1.0 means "as tubular as the most vessel-like voxel in the case's
    # narrow band". On real CTA data this mean, sampled along a traced
    # centreline, typically lands well under 0.3 even for genuine small
    # arteries -- a higher bar here silently rejects every branch. This
    # default is a starting point, not a tuned value: revisit it once
    # reference annotations are available to score precision/recall.
    validated = []
    for region in regions:
        if not check_persistence(region, min_persisted_mm):
            continue
        if not check_direction(region):
            continue
        if region.mean_vesselness < min_vesselness:
            continue
        if not check_continuity(region):
            continue
        validated.append(region)
    return validated
